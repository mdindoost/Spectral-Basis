"""
Q1: Spectral-Label Alignment Analysis
=======================================
Recomputes U from scratch (not stored on disk) using the EXACT same
pipeline as master_training.py. Uses fixed splits throughout.

Per (dataset, k):
  1. Linear probe: LogReg accuracy using first 5/10/20/50 cols of U
  2. Fisher(v1): Fisher criterion on first eigenvector (by eigenvalue)
  3. CCA(U, Y_onehot): canonical correlations between U and label one-hot

Output (PAPER_OUTPUT/q1_analysis/):
  q1_linear_probe_accuracy.csv
  q1_fisher_scores.csv
  q1_canonical_correlations.csv
  plot_q1_linear_probe_vs_k.pdf/.png
  plot_q1_fisher_v1_vs_k.pdf/.png
  q1_summary.txt
"""

import sys, os, warnings, time, json, csv
sys.stdout.reconfigure(line_buffering=True)

# PyTorch 2.6 changed torch.load default to weights_only=True, breaking OGB.
import torch as _torch
_torch_load_orig = _torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load_orig(*args, **kwargs)
_torch.load = _torch_load_compat
import numpy as np
import scipy.sparse as sp
from scipy import stats

# Paths
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
RESULTS_DIR = os.path.join(REPO_ROOT, "PAPER_EXPERIMENTS", "PAPER_RESULTS")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "PAPER_EXPERIMENTS", "PAPER_OUTPUT", "q1_analysis")
DATASET_ROOT = os.path.join(REPO_ROOT, "dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from src.graph_utils import (
    load_dataset, build_graph_matrices,
    get_largest_connected_component_nx, extract_subgraph,
    compute_sgc_normalized_adjacency, sgc_precompute,
    compute_restricted_eigenvectors,
)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ALL_9 = [
    "amazon-computers", "amazon-photo", "citeseer",
    "coauthor-cs", "coauthor-physics", "cora",
    "ogbn-arxiv", "pubmed", "wikics",
]
AVAILABLE_K = [1, 2, 4, 6, 8, 10, 12, 20, 30]
PROBE_SIZES = [5, 10, 20, 50]    # number of leading eigenvectors for linear probe
CROSSOVER_DATASETS = {"amazon-computers", "amazon-photo", "ogbn-arxiv"}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def fisher_criterion(v, labels, train_idx):
    """Fisher criterion for a single vector v using training nodes only.
    Fisher(v) = between_class_variance / within_class_variance
    Formula: Σ_c n_c (μ_c - μ)^2 / Σ_c Σ_{i∈c} (v[i] - μ_c)^2
    """
    v_tr = v[train_idx]
    y_tr = labels[train_idx]
    classes = np.unique(y_tr)
    mu_global = v_tr.mean()

    between = 0.0
    within  = 0.0
    for c in classes:
        mask_c = (y_tr == c)
        v_c = v_tr[mask_c]
        if len(v_c) == 0:
            continue
        mu_c = v_c.mean()
        between += len(v_c) * (mu_c - mu_global) ** 2
        within  += np.sum((v_c - mu_c) ** 2)

    if within < 1e-12:
        return float("nan")
    return float(between / within)


def cca_canonical_correlations(U, Y, max_corr=10):
    """Compute canonical correlations between U (n×p) and Y (n×q).
    Uses QR-based approach: stable even when p >> n.
    Returns array of canonical correlations (length = min(rank_u, rank_y, max_corr)).
    """
    n = U.shape[0]
    # Center
    U_c = U - U.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)

    # QR decompositions — get orthonormal bases for column spaces
    Q_u, _ = np.linalg.qr(U_c)   # n × n
    Q_y, _ = np.linalg.qr(Y_c)   # n × q

    # Trim to numerical rank
    rank_u = np.linalg.matrix_rank(U_c, tol=1e-8)
    rank_y = np.linalg.matrix_rank(Y_c, tol=1e-8)
    Q_u = Q_u[:, :rank_u]
    Q_y = Q_y[:, :rank_y]

    # Canonical correlations = singular values of Q_u.T @ Q_y
    M = Q_u.T @ Q_y
    sv = np.linalg.svd(M, compute_uv=False)
    canonical_corrs = np.clip(sv, 0.0, 1.0)

    n_available = len(canonical_corrs)
    return canonical_corrs, n_available


def load_and_preprocess(dataset_name):
    """Load dataset, extract LCC, return preprocessed data dict.
    Replicates master_training.py pipeline exactly.
    """
    print(f"\n  Loading {dataset_name}...")
    (edge_index, X_raw, labels, num_nodes, num_classes,
     train_idx_orig, val_idx_orig, test_idx_orig) = load_dataset(dataset_name, root=DATASET_ROOT)

    adj, L, D = build_graph_matrices(edge_index, num_nodes)

    # LCC extraction
    lcc_mask = get_largest_connected_component_nx(adj)
    split_idx_original = {
        'train_idx': train_idx_orig,
        'val_idx':   val_idx_orig,
        'test_idx':  test_idx_orig,
    }
    adj, X_raw, labels, split_idx = extract_subgraph(
        adj, X_raw, labels, lcc_mask, split_idx_original
    )

    # Rebuild matrices for LCC (strip self-loops first — same fix as master_training.py)
    adj_no_loops   = adj - sp.diags(adj.diagonal())
    adj_coo        = adj_no_loops.tocoo()
    edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
    adj, L, D      = build_graph_matrices(edge_index_lcc, adj.shape[0])

    num_nodes   = X_raw.shape[0]
    num_classes = len(np.unique(labels))

    # Dense features
    features_dense = X_raw.toarray() if sp.issparse(X_raw) else X_raw.copy()

    # Fixed split
    train_idx = split_idx['train_idx']
    val_idx   = split_idx['val_idx']
    test_idx  = split_idx['test_idx']

    # Normalized adjacency (computed once, reused across k)
    adj_norm = compute_sgc_normalized_adjacency(adj)

    # One-hot label matrix (full graph, used for CCA)
    Y_onehot = np.eye(num_classes)[labels]   # n × C

    print(f"  {dataset_name}: n={num_nodes:,}  C={num_classes}  "
          f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    return dict(
        dataset_name   = dataset_name,
        features_dense = features_dense,
        labels         = labels,
        num_classes    = num_classes,
        num_nodes      = num_nodes,
        adj_norm       = adj_norm,
        L              = L,
        D              = D,
        train_idx      = train_idx,
        val_idx        = val_idx,
        test_idx       = test_idx,
        Y_onehot       = Y_onehot,
    )


# ─── Main loop ────────────────────────────────────────────────────────────────

probe_rows = []     # Q1.1
fisher_rows = []    # Q1.2
cca_rows = []       # Q1.3
verification = []

total_start = time.time()

for dataset_name in ALL_9:
    ds = load_and_preprocess(dataset_name)

    train_idx  = ds['train_idx']
    test_idx   = ds['test_idx']
    labels     = ds['labels']
    num_classes = ds['num_classes']
    Y_onehot   = ds['Y_onehot']

    for k in AVAILABLE_K:
        t0 = time.time()

        # ── SGC diffusion ────────────────────────────────────────────────────
        X_diff = sgc_precompute(ds['features_dense'].copy(), ds['adj_norm'], k)

        # ── Rayleigh-Ritz ────────────────────────────────────────────────────
        U, eigenvalues, d_eff, ortho_error = compute_restricted_eigenvectors(
            X_diff, ds['L'], ds['D'], num_components=0
        )

        # ── Verification ─────────────────────────────────────────────────────
        eig_min = float(eigenvalues.min())
        eig_max = float(eigenvalues.max())
        ok_ortho = ortho_error < 1e-6
        ok_eig   = eig_min >= -0.01 and eig_max <= 2.1
        v_msg = (f"{'OK' if ok_ortho else 'FAIL'}  "
                 f"ortho={ortho_error:.2e}  "
                 f"eig=[{eig_min:.4f},{eig_max:.4f}]  "
                 f"{'OK' if ok_eig else 'FLAG-L/D-SWAP'}")
        verification.append(f"[{dataset_name} k={k}] {v_msg}")

        # ── Q1.1: Linear probe ───────────────────────────────────────────────
        U_train = U[train_idx]
        U_test  = U[test_idx]
        y_train = labels[train_idx]
        y_test  = labels[test_idx]

        for p in PROBE_SIZES:
            p_actual = min(p, d_eff)
            U_tr_p = U_train[:, :p_actual]
            U_te_p = U_test[:,  :p_actual]

            # Standardize (important for convergence)
            scaler = StandardScaler()
            U_tr_p = scaler.fit_transform(U_tr_p)
            U_te_p = scaler.transform(U_te_p)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(max_iter=1000, random_state=0,
                                         solver='lbfgs', multi_class='auto')
                clf.fit(U_tr_p, y_train)
            acc = float(clf.score(U_te_p, y_test)) * 100

            probe_rows.append(dict(
                dataset               = dataset_name,
                k                     = k,
                num_eigenvectors_used = p_actual,
                requested_p           = p,
                d_eff                 = d_eff,
                linear_probe_accuracy = round(acc, 4),
            ))

        # Also run with all eigenvectors if d_eff > 50
        if d_eff > 50:
            scaler = StandardScaler()
            U_tr_all = scaler.fit_transform(U_train)
            U_te_all = scaler.transform(U_test)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(max_iter=1000, random_state=0,
                                         solver='lbfgs', multi_class='auto')
                clf.fit(U_tr_all, y_train)
            acc_all = float(clf.score(U_te_all, y_test)) * 100
            probe_rows.append(dict(
                dataset               = dataset_name,
                k                     = k,
                num_eigenvectors_used = d_eff,
                requested_p           = "all",
                d_eff                 = d_eff,
                linear_probe_accuracy = round(acc_all, 4),
            ))

        # ── Q1.2: Fisher(v₁) — first eigenvector by eigenvalue ──────────────
        # Eigenvalues already sorted ascending; U[:,0] = lowest-eigenvalue eigenvector
        v1 = U[:, 0]
        fisher_v1 = fisher_criterion(v1, labels, train_idx)

        # Also compute for top-5 eigenvectors to show trend
        fisher_per_col = []
        for j in range(min(20, d_eff)):
            fisher_per_col.append(fisher_criterion(U[:, j], labels, train_idx))

        fisher_rows.append(dict(
            dataset           = dataset_name,
            k                 = k,
            d_eff             = d_eff,
            fisher_v1         = round(fisher_v1, 6) if not np.isnan(fisher_v1) else "NaN",
            fisher_v2         = round(fisher_per_col[1], 6) if len(fisher_per_col) > 1 else "N/A",
            fisher_v3         = round(fisher_per_col[2], 6) if len(fisher_per_col) > 2 else "N/A",
            fisher_v5         = round(fisher_per_col[4], 6) if len(fisher_per_col) > 4 else "N/A",
            fisher_v10        = round(fisher_per_col[9], 6) if len(fisher_per_col) > 9 else "N/A",
            fisher_v20        = round(fisher_per_col[19], 6) if len(fisher_per_col) > 19 else "N/A",
            fisher_mean_top20 = round(float(np.nanmean(fisher_per_col)), 6),
            fisher_max_top20  = round(float(np.nanmax(fisher_per_col)), 6),
            eig_v1            = round(float(eigenvalues[0]), 6),
        ))

        # ── Q1.3: CCA(U_train, Y_train_onehot) ──────────────────────────────
        U_tr_cca = U[train_idx]
        Y_tr_cca = Y_onehot[train_idx]

        try:
            canonical_corrs, n_avail = cca_canonical_correlations(U_tr_cca, Y_tr_cca)

            def mean_top(arr, n):
                if len(arr) >= n:
                    return round(float(arr[:n].mean()), 6)
                elif len(arr) > 0:
                    return round(float(arr.mean()), 6)
                return "N/A"

            cca_rows.append(dict(
                dataset              = dataset_name,
                k                    = k,
                d_eff                = d_eff,
                num_classes          = num_classes,
                n_canonical_corrs    = n_avail,
                top_1_canonical_corr = round(float(canonical_corrs[0]), 6) if n_avail > 0 else "N/A",
                top_5_canonical_corr = mean_top(canonical_corrs, 5),
                top_10_canonical_corr= mean_top(canonical_corrs, 10),
                mean_canonical_corr  = round(float(canonical_corrs.mean()), 6) if n_avail > 0 else "N/A",
            ))
        except Exception as e:
            cca_rows.append(dict(
                dataset="ERROR", k=k, d_eff=d_eff, num_classes=num_classes,
                n_canonical_corrs=0,
                top_1_canonical_corr="ERR", top_5_canonical_corr="ERR",
                top_10_canonical_corr="ERR", mean_canonical_corr="ERR",
            ))
            verification.append(f"  CCA ERROR [{dataset_name} k={k}]: {e}")

        elapsed = time.time() - t0
        print(f"  k={k:>2d}  d_eff={d_eff:>5d}  "
              f"ortho={ortho_error:.1e}  "
              f"probe@50={next((r['linear_probe_accuracy'] for r in probe_rows[::-1] if r['dataset']==dataset_name and r['k']==k and r['requested_p']==50), '?'):>6.2f}%  "
              f"Fisher(v1)={fisher_v1:.4f}  "
              f"CCA_top5={cca_rows[-1]['top_5_canonical_corr']}  "
              f"({elapsed:.1f}s)")

# ─── Write CSVs ──────────────────────────────────────────────────────────────

probe_csv = os.path.join(OUTPUT_DIR, "q1_linear_probe_accuracy.csv")
with open(probe_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "dataset", "k", "num_eigenvectors_used", "requested_p",
        "d_eff", "linear_probe_accuracy"
    ])
    writer.writeheader(); writer.writerows(probe_rows)
print(f"\nSaved: {probe_csv}")

fisher_csv = os.path.join(OUTPUT_DIR, "q1_fisher_scores.csv")
with open(fisher_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "dataset", "k", "d_eff", "fisher_v1", "fisher_v2", "fisher_v3",
        "fisher_v5", "fisher_v10", "fisher_v20",
        "fisher_mean_top20", "fisher_max_top20", "eig_v1"
    ])
    writer.writeheader(); writer.writerows(fisher_rows)
print(f"Saved: {fisher_csv}")

cca_csv = os.path.join(OUTPUT_DIR, "q1_canonical_correlations.csv")
with open(cca_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "dataset", "k", "d_eff", "num_classes", "n_canonical_corrs",
        "top_1_canonical_corr", "top_5_canonical_corr",
        "top_10_canonical_corr", "mean_canonical_corr"
    ])
    writer.writeheader(); writer.writerows(cca_rows)
print(f"Saved: {cca_csv}")

# ─── PLOT 1: Linear probe accuracy vs k ──────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS_9 = {
    "amazon-computers":  "#e03030",
    "amazon-photo":      "#e07020",
    "citeseer":          "#808080",
    "coauthor-cs":       "#d4a800",
    "coauthor-physics":  "#9050c0",
    "cora":              "#2060c0",
    "ogbn-arxiv":        "#c040c0",
    "pubmed":            "#106090",
    "wikics":            "#208050",
}

# One subplot per probe size
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)

for ax_i, p in enumerate(PROBE_SIZES):
    ax = axes[ax_i]
    for ds in ALL_9:
        rows = sorted(
            [r for r in probe_rows if r['dataset'] == ds and r['requested_p'] == p],
            key=lambda r: r['k']
        )
        if not rows: continue
        ks   = [r['k'] for r in rows]
        accs = [r['linear_probe_accuracy'] for r in rows]
        lw   = 2.2 if ds in CROSSOVER_DATASETS else 1.4
        ax.plot(ks, accs, color=COLORS_9[ds], linewidth=lw,
                marker='o', markersize=3.5,
                linestyle='-',
                label=ds + (' ✕' if ds in CROSSOVER_DATASETS else ''))
    ax.set_title(f"First {p} eigenvectors", fontsize=10)
    ax.set_xlabel("k", fontsize=9)
    if ax_i == 0:
        ax.set_ylabel("Linear probe accuracy (%)", fontsize=9)
    ax.set_xticks([1, 4, 10, 20, 30])
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.3)

# One shared legend
handles, labels_leg = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc='lower center', ncol=5,
           fontsize=8, bbox_to_anchor=(0.5, -0.05), framealpha=0.9)
fig.suptitle("Q1.1: Linear Probe Accuracy on First p Eigenvectors of U  (fixed splits)\n"
             "✕ = crossover dataset", fontsize=11)
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(OUTPUT_DIR, f'plot_q1_linear_probe_vs_k.{ext}'),
                dpi=150, bbox_inches='tight')
plt.close()

# ─── PLOT 2: Fisher(v₁) vs k ─────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5.5))
for ds in ALL_9:
    rows = sorted([r for r in fisher_rows if r['dataset'] == ds], key=lambda r: r['k'])
    if not rows: continue
    ks = [r['k'] for r in rows]
    fv = []
    for r in rows:
        val = r['fisher_v1']
        fv.append(float(val) if val != 'NaN' else float('nan'))
    lw = 2.2 if ds in CROSSOVER_DATASETS else 1.4
    ax2.plot(ks, fv, color=COLORS_9[ds], linewidth=lw,
             marker='o', markersize=4,
             label=ds + (' ✕' if ds in CROSSOVER_DATASETS else ''))

ax2.set_xlabel("Diffusion depth k", fontsize=12)
ax2.set_ylabel("Fisher(v₁)  [between/within class variance]", fontsize=11)
ax2.set_title("Q1.2: Fisher Score of First Eigenvector v₁ vs k  (fixed splits)\n"
              "✕ = crossover dataset", fontsize=11)
ax2.set_xticks(AVAILABLE_K)
ax2.legend(fontsize=8, ncol=2, loc='upper right', framealpha=0.85)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
for ext in ['pdf', 'png']:
    fig2.savefig(os.path.join(OUTPUT_DIR, f'plot_q1_fisher_v1_vs_k.{ext}'),
                 dpi=150, bbox_inches='tight')
plt.close()

# ─── Summary ─────────────────────────────────────────────────────────────────
fails = [l for l in verification if 'FAIL' in l or 'FLAG' in l or 'ERROR' in l]
ok    = [l for l in verification if l.startswith('[') and 'FAIL' not in l and 'FLAG' not in l]

print(f"\n{'='*70}")
print("Q1 VERIFICATION SUMMARY")
print(f"  D-orthonormality checks passed: {len(ok)}")
print(f"  Failures/flags: {len(fails)}")
for l in fails:
    print(f"    {l}")

print(f"\n{'='*70}")
print("Q1.1 LINEAR PROBE — accuracy at k=10 using first 50 eigenvectors")
print(f"  {'Dataset':25s}  {'p=5':>8}  {'p=10':>8}  {'p=20':>8}  {'p=50':>8}  {'p=all':>8}")
print("  " + "-"*65)
for ds in ALL_9:
    def gp(p_val):
        r = next((r for r in probe_rows if r['dataset']==ds
                  and r['k']==10 and r['requested_p']==p_val), None)
        return f"{r['linear_probe_accuracy']:.1f}%" if r else " N/A  "
    print(f"  {ds:25s}  {gp(5):>8}  {gp(10):>8}  {gp(20):>8}  {gp(50):>8}  {gp('all'):>8}")

print(f"\n{'='*70}")
print("Q1.2 FISHER(v₁) at k=[1, 10, 30]")
print(f"  {'Dataset':25s}  {'k=1':>10}  {'k=10':>10}  {'k=30':>10}")
print("  " + "-"*58)
for ds in ALL_9:
    def gf(k_val):
        r = next((r for r in fisher_rows if r['dataset']==ds and r['k']==k_val), None)
        return f"{r['fisher_v1']}" if r else "N/A"
    print(f"  {ds:25s}  {gf(1):>10}  {gf(10):>10}  {gf(30):>10}")

print(f"\n{'='*70}")
print("Q1.3 CCA(U_train, Y_labels) — top-5 canonical corr at k=[1, 10, 30]")
print(f"  {'Dataset':25s}  {'k=1':>10}  {'k=10':>10}  {'k=30':>10}")
print("  " + "-"*58)
for ds in ALL_9:
    def gc(k_val):
        r = next((r for r in cca_rows if r['dataset']==ds and r['k']==k_val), None)
        return f"{r['top_5_canonical_corr']}" if r else "N/A"
    print(f"  {ds:25s}  {gc(1):>10}  {gc(10):>10}  {gc(30):>10}")

# Save summary
summary_path = os.path.join(OUTPUT_DIR, "q1_summary.txt")
with open(summary_path, "w") as f:
    f.write("Q1: SPECTRAL-LABEL ALIGNMENT ANALYSIS\n")
    f.write("="*60 + "\n")
    f.write(f"Verification: {len(ok)} passed, {len(fails)} failed\n")
    f.write("\n".join(verification))
print(f"\nSaved: {summary_path}")
print(f"\nTotal runtime: {(time.time()-total_start)/60:.1f} min")
print(f"Done. All outputs in: {OUTPUT_DIR}")
