"""
Fisher(v₁) vs Part A for Crossover Datasets
=============================================
Recomputes U for amazon-computers, amazon-photo, ogbn-arxiv at all k.
Extracts Fisher score of first eigenvector (lowest eigenvalue direction).
Compares with Part A (loaded from existing PAPER_RESULTS).

Outputs:
  PAPER_OUTPUT/fisher_crossover/fisher_crossover_analysis.csv
  PAPER_OUTPUT/fisher_crossover/fisher_vs_part_a_crossover_datasets.png/.pdf
"""

import sys, os, warnings, time
import numpy as np
import scipy.sparse as sp
import csv

sys.stdout.reconfigure(line_buffering=True)

# PyTorch 2.6 / OGB compatibility fix
import torch as _torch
_torch_load_orig = _torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load_orig(*args, **kwargs)
_torch.load = _torch_load_compat

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
RESULTS_DIR  = os.path.join(REPO_ROOT, "PAPER_EXPERIMENTS", "PAPER_RESULTS")
DATASET_ROOT = os.path.join(REPO_ROOT, "dataset")
OUTPUT_DIR   = os.path.join(REPO_ROOT, "PAPER_EXPERIMENTS", "PAPER_OUTPUT", "fisher_crossover")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from src.graph_utils import (
    load_dataset, build_graph_matrices,
    get_largest_connected_component_nx, extract_subgraph,
    compute_sgc_normalized_adjacency, sgc_precompute,
    compute_restricted_eigenvectors,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Config ───────────────────────────────────────────────────────────────────
DATASETS    = ["amazon-computers", "amazon-photo", "ogbn-arxiv"]
AVAILABLE_K = [1, 2, 4, 6, 8, 10, 12, 20, 30]
SPLIT       = "fixed"
FISHER_THRESHOLD = 0.1

COLORS = {
    "amazon-computers": "#e03030",
    "amazon-photo":     "#e07020",
    "ogbn-arxiv":       "#c040c0",
}

# Known crossover k values from Q2
K_CROSSOVER = {
    "amazon-computers": 5.08,
    "amazon-photo":     8.18,
    "ogbn-arxiv":       6.23,
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def fisher_v1(U, labels, train_idx):
    """Fisher criterion for first eigenvector column (training nodes only)."""
    v = U[train_idx, 0]
    y = labels[train_idx]
    classes = np.unique(y)
    mu = v.mean()
    between, within = 0.0, 0.0
    for c in classes:
        vc = v[y == c]
        if len(vc) == 0: continue
        mu_c = vc.mean()
        between += len(vc) * (mu_c - mu) ** 2
        within  += np.sum((vc - mu_c) ** 2)
    return float(between / within) if within > 1e-12 else float('nan')

def load_part_a(dataset, k):
    """Load Part A from existing PAPER_RESULTS metrics JSON."""
    path = os.path.join(RESULTS_DIR, f"{dataset}_{SPLIT}_lcc",
                        f"k{k}", "metrics", "results.json")
    if not os.path.exists(path):
        return None
    import json
    d = json.load(open(path))
    return d.get("framework_analysis", {}).get("part_a_pp")

def find_threshold_crossing(ks, values, threshold):
    """Linear interpolation of k where value crosses threshold (downward)."""
    for i in range(len(values) - 1):
        v0, v1 = values[i], values[i+1]
        if v0 >= threshold > v1:      # downward crossing
            k0, k1 = ks[i], ks[i+1]
            k_cross = k0 + (threshold - v0) * (k1 - k0) / (v1 - v0)
            return k_cross
        elif v0 <= threshold < v1:    # upward crossing (unexpected but handle)
            k0, k1 = ks[i], ks[i+1]
            k_cross = k0 + (threshold - v0) * (k1 - k0) / (v1 - v0)
            return k_cross
    return None

def load_and_preprocess(dataset_name):
    """Exact same pipeline as master_training.py."""
    print(f"\n  Loading {dataset_name}...")
    (edge_index, X_raw, labels, num_nodes, num_classes,
     tr_orig, va_orig, te_orig) = load_dataset(dataset_name, root=DATASET_ROOT)

    adj, L, D = build_graph_matrices(edge_index, num_nodes)
    lcc_mask  = get_largest_connected_component_nx(adj)

    adj, X_raw, labels, split_idx = extract_subgraph(
        adj, X_raw, labels, lcc_mask,
        {'train_idx': tr_orig, 'val_idx': va_orig, 'test_idx': te_orig}
    )
    adj_nl   = adj - sp.diags(adj.diagonal())
    ei_lcc   = np.vstack([adj_nl.tocoo().row, adj_nl.tocoo().col])
    adj, L, D = build_graph_matrices(ei_lcc, adj.shape[0])

    features = X_raw.toarray() if sp.issparse(X_raw) else X_raw.copy()
    adj_norm = compute_sgc_normalized_adjacency(adj)
    train_idx = split_idx['train_idx']

    print(f"  n={adj.shape[0]:,}  train={len(train_idx)}  classes={len(np.unique(labels))}")
    return dict(features=features, labels=labels, adj_norm=adj_norm,
                L=L, D=D, train_idx=train_idx)

# ─── Main computation ─────────────────────────────────────────────────────────
print("=" * 60)
print("Fisher(v₁) vs Part A — Crossover Datasets")
print("=" * 60)

all_results = {}   # dataset -> list of dicts

for ds in DATASETS:
    data = load_and_preprocess(ds)
    results = []

    print(f"\n  k sweep for {ds}:")
    print(f"  {'k':>4}  {'d_eff':>7}  {'ortho':>10}  {'Fisher(v1)':>12}  {'Part A':>9}")
    print("  " + "-" * 55)

    for k in AVAILABLE_K:
        t0 = time.time()
        X_diff = sgc_precompute(data['features'].copy(), data['adj_norm'], k)
        U, eigenvalues, d_eff, ortho_err = compute_restricted_eigenvectors(
            X_diff, data['L'], data['D'], num_components=0
        )

        fv1    = fisher_v1(U, data['labels'], data['train_idx'])
        part_a = load_part_a(ds, k)

        results.append(dict(
            dataset    = ds,
            k          = k,
            d_eff      = d_eff,
            ortho_error= ortho_err,
            fisher_v1  = round(fv1, 6) if not np.isnan(fv1) else None,
            part_a_pp  = round(part_a, 4) if part_a is not None else None,
            eig_v1     = round(float(eigenvalues[0]), 6),
        ))

        print(f"  {k:>4}  {d_eff:>7}  {ortho_err:>10.2e}  "
              f"{fv1:>12.6f}  "
              f"{part_a:>+9.2f}pp  ({time.time()-t0:.1f}s)")

    all_results[ds] = results

# ─── Threshold analysis ────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"THRESHOLD ANALYSIS  (Fisher(v₁) < {FISHER_THRESHOLD})")
print("=" * 60)

csv_rows = []

for ds in DATASETS:
    res = all_results[ds]
    ks       = [r['k'] for r in res if r['fisher_v1'] is not None]
    fishers  = [r['fisher_v1'] for r in res if r['fisher_v1'] is not None]
    part_as  = [r['part_a_pp'] for r in res if r['part_a_pp'] is not None]

    k_fisher_thresh = find_threshold_crossing(ks, fishers, FISHER_THRESHOLD)
    k_part_a_zero   = K_CROSSOVER[ds]   # from Q2

    fisher_at_k10 = next((r['fisher_v1'] for r in res if r['k'] == 10), None)
    fisher_at_k1  = next((r['fisher_v1'] for r in res if r['k'] == 1),  None)
    part_a_at_k1  = next((r['part_a_pp'] for r in res if r['k'] == 1),  None)

    lag = None
    if k_fisher_thresh is not None:
        lag = round(k_fisher_thresh - k_part_a_zero, 2)
        direction = "Fisher drops AFTER Part A crosses" if lag > 0 else "Fisher drops BEFORE Part A crosses"
        print(f"\n  {ds}:")
        print(f"    Part A crosses 0 at    k ≈ {k_part_a_zero:.1f}")
        print(f"    Fisher(v₁) < {FISHER_THRESHOLD} at    k ≈ {k_fisher_thresh:.1f}")
        print(f"    Lag (Fisher - PartA):  Δk = {lag:+.1f}  → {direction}")
    else:
        print(f"\n  {ds}: Fisher(v₁) never drops below {FISHER_THRESHOLD} in k={AVAILABLE_K}")
        print(f"    Fisher range: [{min(fishers):.4f}, {max(fishers):.4f}]")
        print(f"    Part A crosses 0 at k ≈ {k_part_a_zero:.1f}")

    for r in res:
        csv_rows.append(dict(
            dataset               = ds,
            k                     = r['k'],
            d_eff                 = r['d_eff'],
            fisher_v1             = r['fisher_v1'],
            part_a_pp             = r['part_a_pp'],
            eig_v1                = r['eig_v1'],
            ortho_error           = r['ortho_error'],
            k_part_a_crossover    = k_part_a_zero,
            k_fisher_threshold    = round(k_fisher_thresh, 2) if k_fisher_thresh else "N/A",
            fisher_threshold_used = FISHER_THRESHOLD,
            lag_fisher_minus_parta= lag if lag is not None else "N/A",
        ))

# ─── Save CSV ─────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "fisher_crossover_analysis.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\nSaved: {csv_path}")

# ─── Plot: 3-panel dual-axis ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax_i, ds in enumerate(DATASETS):
    ax   = axes[ax_i]
    ax2  = ax.twinx()
    res  = all_results[ds]
    color = COLORS[ds]

    ks      = [r['k'] for r in res]
    fishers = [r['fisher_v1'] if r['fisher_v1'] is not None else float('nan') for r in res]
    part_as = [r['part_a_pp'] if r['part_a_pp'] is not None else float('nan') for r in res]

    # Left axis: Fisher(v₁)
    l1, = ax.plot(ks, fishers, color="#2277cc", linewidth=2.2,
                  marker="s", markersize=5, label="Fisher(v₁)", zorder=3)
    ax.fill_between(ks, fishers, alpha=0.10, color="#2277cc")

    # Threshold line
    ax.axhline(FISHER_THRESHOLD, color="#2277cc", linewidth=1.0,
               linestyle=":", alpha=0.7, label=f"threshold={FISHER_THRESHOLD}")

    # Right axis: Part A
    l2, = ax2.plot(ks, part_as, color=color, linewidth=2.2,
                   marker="o", markersize=5, label="Part A [pp]", zorder=3)
    ax2.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.5)

    # Mark Part A crossover
    k_zero = K_CROSSOVER[ds]
    ax2.axvline(k_zero, color=color, linewidth=1.2, linestyle="--",
                alpha=0.7, zorder=1, label=f"Part A=0 k≈{k_zero:.1f}")

    # Mark Fisher threshold crossing
    valid_k = [r['k'] for r in res if r['fisher_v1'] is not None]
    valid_f = [r['fisher_v1'] for r in res if r['fisher_v1'] is not None]
    k_fth = find_threshold_crossing(valid_k, valid_f, FISHER_THRESHOLD)
    if k_fth is not None:
        ax.axvline(k_fth, color="#2277cc", linewidth=1.2, linestyle=":",
                   alpha=0.8, zorder=1)
        ax.annotate(f"Fisher<{FISHER_THRESHOLD}\nk≈{k_fth:.1f}",
                    xy=(k_fth, FISHER_THRESHOLD),
                    xytext=(k_fth + 1.5, FISHER_THRESHOLD + 0.05),
                    fontsize=7.5, color="#2277cc",
                    arrowprops=dict(arrowstyle="-", color="#2277cc", lw=0.8))

    # Axis formatting
    ax.set_xlabel("Diffusion depth k", fontsize=10)
    ax.set_ylabel("Fisher(v₁)", color="#2277cc", fontsize=10)
    ax.tick_params(axis="y", labelcolor="#2277cc")
    ax2.set_ylabel("Part A [pp]", color=color, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=color)
    ax.set_xticks([1, 4, 8, 12, 20, 30])
    ax.set_title(f"{ds}\n[Crossover at k≈{k_zero:.1f}]",
                 fontsize=10, color=color, fontweight="bold")
    ax.grid(True, alpha=0.25)

    # Legend
    lines = [l1, l2]
    labels_leg = [l.get_label() for l in lines]
    ax.legend(lines, labels_leg, fontsize=8, loc="upper right", framealpha=0.85)

fig.suptitle(
    f"Fisher(v₁) vs Part A — Crossover Datasets  (fixed splits)\n"
    f"Blue=Fisher score of lowest eigenvector  |  Colored=Part A gap  |  "
    f"Threshold={FISHER_THRESHOLD}",
    fontsize=11
)
plt.tight_layout(rect=[0, 0, 1, 0.92])

for ext in ["png", "pdf"]:
    p = os.path.join(OUTPUT_DIR, f"fisher_vs_part_a_crossover_datasets.{ext}")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved: {p}")
plt.close()

print("\nDone.")
