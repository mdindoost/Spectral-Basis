"""
Q3: Graph Topology Features for Crossover Prediction
======================================================
Loads each dataset ONCE, extracts LCC, computes structural properties.
No training, no eigenvectors — purely graph topology.
Runs once per dataset (topology doesn't change with k).

Metrics computed:
  - avg clustering coefficient
  - transitivity (global clustering)
  - avg shortest path length (sampled if graph > 5000 nodes)
  - effective diameter (90th percentile of sampled distances)
  - spectral gap (lambda_2 - lambda_1 of full normalized Laplacian)
  - algebraic connectivity (lambda_2, Fiedler value)
  - eigenvalue ratio (lambda_max / lambda_2)
  - modularity of ground-truth label partition
  - label assortativity coefficient

Outputs (PAPER_OUTPUT/q3_analysis/):
  q3_graph_properties.csv
  plot_q3_crossover_vs_monotonic.pdf/.png   (bar chart comparison)
  plot_q3_property_profiles.pdf/.png        (radar/spider chart)
  q3_summary.txt
"""

import sys, os, time, csv, warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx

sys.stdout.reconfigure(line_buffering=True)

# PyTorch 2.6 changed torch.load default to weights_only=True, breaking OGB.
# Monkey-patch to restore the old behaviour for trusted local files.
import torch as _torch
_torch_load_orig = _torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load_orig(*args, **kwargs)
_torch.load = _torch_load_compat

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
DATASET_ROOT = os.path.join(REPO_ROOT, "dataset")
OUTPUT_DIR   = os.path.join(REPO_ROOT, "PAPER_EXPERIMENTS", "PAPER_OUTPUT", "q3_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from src.graph_utils import (
    load_dataset, build_graph_matrices,
    get_largest_connected_component_nx, extract_subgraph,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Dataset config ───────────────────────────────────────────────────────────
ALL_9 = [
    "amazon-computers", "amazon-photo", "citeseer",
    "coauthor-cs", "coauthor-physics", "cora",
    "ogbn-arxiv", "pubmed", "wikics",
]

# From Q2 results
CROSSOVER_TYPES = {
    "amazon-computers":  "Crossover",
    "amazon-photo":      "Crossover",
    "ogbn-arxiv":        "Crossover",
    "wikics":            "Near-crossover",
    "citeseer":          "Monotonic",
    "coauthor-cs":       "Monotonic",
    "coauthor-physics":  "Near-crossover",
    "cora":              "Monotonic",
    "pubmed":            "Monotonic",
}

# Part A at k=10 from earlier analysis
PART_A_K10 = {
    "amazon-computers": -5.55,
    "amazon-photo":     -0.62,
    "citeseer":         36.57,
    "coauthor-cs":      13.06,
    "coauthor-physics":  8.02,
    "cora":             29.89,
    "ogbn-arxiv":       -0.68,
    "pubmed":           27.57,
    "wikics":            8.63,
}

# Sampling budget for expensive metrics on large graphs
SAMPLE_NODES_PATH = 1000   # pairs for avg shortest path
SAMPLE_SEED       = 42


# ─── Metric computations ──────────────────────────────────────────────────────

def compute_clustering(G):
    """Average clustering coefficient (node-level average)."""
    return nx.average_clustering(G)


def compute_transitivity(G):
    """Global clustering coefficient (triangle-based)."""
    return nx.transitivity(G)


def compute_path_stats(G, n_nodes, sample_size=SAMPLE_NODES_PATH, seed=SAMPLE_SEED):
    """Avg shortest path length and effective diameter (90th pct).
    For large graphs: sample random pairs; for small: exact.
    Returns (avg_path_length, effective_diameter, method)
    """
    rng = np.random.default_rng(seed)

    if n_nodes <= 2000:
        # Exact computation
        try:
            avg = nx.average_shortest_path_length(G)
            # Diameter: sample for speed even on small graphs
            all_lengths = []
            for source in list(G.nodes())[:200]:
                lengths = nx.single_source_shortest_path_length(G, source)
                all_lengths.extend(lengths.values())
            eff_diam = float(np.percentile(all_lengths, 90))
            return avg, eff_diam, "exact"
        except Exception:
            pass

    # Sampled computation
    nodes = list(G.nodes())
    sources = rng.choice(nodes, size=min(sample_size, len(nodes)), replace=False)

    all_lengths = []
    for s in sources:
        lengths = nx.single_source_shortest_path_length(G, s)
        all_lengths.extend(lengths.values())

    avg_path = float(np.mean(all_lengths))
    eff_diam = float(np.percentile(all_lengths, 90))
    return avg_path, eff_diam, f"sampled({len(sources)}nodes)"


def compute_spectral_gap(L_sparse, n_nodes, k_eigs=10):
    """Compute first k eigenvalues of the normalized Laplacian.
    Returns: lambda_1, lambda_2 (Fiedler), lambda_max, spectral_gap.
    Uses FULL graph Laplacian (not restricted).
    """
    # Normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}
    # We'll use the sparse L directly and compute its first k eigenvalues
    try:
        # Find smallest k eigenvalues
        k_use = min(k_eigs, n_nodes - 2)
        # sigma='SM' finds smallest magnitude eigenvalues
        eigenvalues, _ = spla.eigsh(L_sparse, k=k_use, which='SM',
                                     tol=1e-6, maxiter=5000)
        eigenvalues = np.sort(np.real(eigenvalues))
        eigenvalues = np.clip(eigenvalues, 0, None)  # fix tiny negatives

        lambda_1  = float(eigenvalues[0])    # should be ~0 for connected LCC
        lambda_2  = float(eigenvalues[1])    # Fiedler value (algebraic connectivity)

        # Also get largest eigenvalue (use 'LM')
        k_large = min(3, n_nodes - 2)
        eig_large, _ = spla.eigsh(L_sparse, k=k_large, which='LM',
                                   tol=1e-6, maxiter=5000)
        lambda_max = float(np.max(np.real(eig_large)))

        spectral_gap = lambda_2 - lambda_1
        eig_ratio    = lambda_max / lambda_2 if lambda_2 > 1e-10 else float('nan')

        return lambda_1, lambda_2, lambda_max, spectral_gap, eig_ratio

    except Exception as e:
        print(f"    WARNING: spectral gap computation failed: {e}")
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')


def compute_normalized_laplacian(adj_sparse):
    """L_norm = D^{-1/2} (D - A) D^{-1/2} = I - D^{-1/2} A D^{-1/2}"""
    deg = np.array(adj_sparse.sum(axis=1)).ravel()
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    # Degree matrix
    D_mat = sp.diags(deg)
    # Laplacian
    L = D_mat - adj_sparse
    # Normalized Laplacian
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    return L_norm


def compute_modularity(G, labels):
    """Modularity of the ground-truth label partition.
    Partition = group nodes by class label.
    """
    classes = np.unique(labels)
    partition = [set(np.where(labels == c)[0]) for c in classes]
    # Filter to nodes actually in G
    g_nodes = set(G.nodes())
    partition = [p & g_nodes for p in partition]
    partition = [p for p in partition if len(p) > 0]

    try:
        mod = nx.community.modularity(G, partition)
        return float(mod)
    except Exception as e:
        print(f"    WARNING: modularity failed: {e}")
        return float('nan')


def compute_label_assortativity(G, labels):
    """Label assortativity coefficient.
    Measures tendency of connected nodes to share the same label.
    """
    # Assign label attribute to nodes
    label_dict = {node: int(labels[node]) for node in G.nodes()
                  if node < len(labels)}
    nx.set_node_attributes(G, label_dict, 'label')
    try:
        r = nx.attribute_assortativity_coefficient(G, 'label')
        return float(r)
    except Exception as e:
        print(f"    WARNING: assortativity failed: {e}")
        return float('nan')


# ─── Main loop ────────────────────────────────────────────────────────────────

rows = []
total_start = time.time()

for dataset_name in ALL_9:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  {dataset_name}  [{CROSSOVER_TYPES[dataset_name]}]")
    print(f"{'='*60}")

    # Load and preprocess (same pipeline as Q1 / master_training.py)
    print("  Loading dataset...")
    (edge_index, X_raw, labels, num_nodes, num_classes,
     train_idx_orig, val_idx_orig, test_idx_orig) = load_dataset(dataset_name, root=DATASET_ROOT)

    adj, L_full, D_full = build_graph_matrices(edge_index, num_nodes)

    lcc_mask = get_largest_connected_component_nx(adj)
    split_idx_orig = {'train_idx': train_idx_orig,
                      'val_idx':   val_idx_orig,
                      'test_idx':  test_idx_orig}
    adj_lcc, X_lcc, labels_lcc, _ = extract_subgraph(
        adj, X_raw, labels, lcc_mask, split_idx_orig
    )

    # Strip self-loops for the LCC adjacency (for graph-theoretic measures)
    adj_no_loops = adj_lcc - sp.diags(adj_lcc.diagonal())
    adj_no_loops = adj_no_loops.tocsr()
    n_lcc = adj_lcc.shape[0]
    n_edges = int(adj_no_loops.nnz) // 2   # undirected

    print(f"  LCC: n={n_lcc:,}  edges={n_edges:,}  classes={num_classes}")

    # Build networkx graph from LCC adjacency (no self-loops)
    print("  Building NetworkX graph...")
    G = nx.from_scipy_sparse_array(adj_no_loops)
    avg_degree = float(np.array(adj_no_loops.sum(1)).ravel().mean())

    # ── Clustering ───────────────────────────────────────────────────────────
    print("  Computing clustering coefficients...")
    t1 = time.time()
    avg_clust = compute_clustering(G)
    transit   = compute_transitivity(G)
    print(f"  avg_clustering={avg_clust:.4f}  transitivity={transit:.4f}  ({time.time()-t1:.1f}s)")

    # ── Path length (sampled for large graphs) ────────────────────────────────
    print("  Computing path lengths (sampled)...")
    t1 = time.time()
    avg_path, eff_diam, path_method = compute_path_stats(G, n_lcc)
    print(f"  avg_path={avg_path:.3f}  eff_diam={eff_diam:.1f}  method={path_method}  ({time.time()-t1:.1f}s)")

    # ── Spectral gap (full graph normalized Laplacian) ────────────────────────
    print("  Computing spectral gap of full normalized Laplacian...")
    t1 = time.time()
    L_norm = compute_normalized_laplacian(adj_no_loops)
    lambda1, lambda2, lambda_max, spec_gap, eig_ratio = compute_spectral_gap(L_norm, n_lcc)
    print(f"  lambda_2(Fiedler)={lambda2:.6f}  spectral_gap={spec_gap:.6f}  "
          f"lambda_max={lambda_max:.4f}  eig_ratio={eig_ratio:.2f}  ({time.time()-t1:.1f}s)")

    # ── Modularity ────────────────────────────────────────────────────────────
    print("  Computing modularity of label partition...")
    t1 = time.time()
    mod = compute_modularity(G, labels_lcc)
    print(f"  modularity={mod:.4f}  ({time.time()-t1:.1f}s)")

    # ── Label assortativity ───────────────────────────────────────────────────
    print("  Computing label assortativity...")
    t1 = time.time()
    assort = compute_label_assortativity(G, labels_lcc)
    print(f"  label_assortativity={assort:.4f}  ({time.time()-t1:.1f}s)")

    # ── Homophily (edge fraction where src==dst label) ────────────────────────
    # Direct from adjacency — faster than networkx
    adj_coo = adj_no_loops.tocoo()
    same_label = (labels_lcc[adj_coo.row] == labels_lcc[adj_coo.col])
    homophily = float(same_label.mean())
    print(f"  homophily={homophily:.4f}")

    elapsed = time.time() - t0
    print(f"  Total for {dataset_name}: {elapsed:.1f}s")

    row = dict(
        dataset              = dataset_name,
        crossover_type       = CROSSOVER_TYPES[dataset_name],
        part_a_k10           = PART_A_K10[dataset_name],
        n_nodes              = n_lcc,
        n_edges              = n_edges,
        num_classes          = num_classes,
        avg_degree           = round(avg_degree, 4),
        avg_clustering       = round(avg_clust, 6),
        transitivity         = round(transit, 6),
        avg_path_length      = round(avg_path, 4),
        effective_diameter   = round(eff_diam, 2),
        path_method          = path_method,
        lambda_1             = round(lambda1, 8),
        lambda_2_fiedler     = round(lambda2, 8),
        lambda_max           = round(lambda_max, 6),
        spectral_gap         = round(spec_gap, 8),
        eigenvalue_ratio     = round(eig_ratio, 4) if not np.isnan(eig_ratio) else "NaN",
        modularity           = round(mod, 6) if not np.isnan(mod) else "NaN",
        label_assortativity  = round(assort, 6) if not np.isnan(assort) else "NaN",
        homophily            = round(homophily, 6),
    )
    rows.append(row)

# ─── Write CSV ────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "q3_graph_properties.csv")
fieldnames = list(rows[0].keys())
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved: {csv_path}")

# ─── Analysis: Crossover vs Monotonic ─────────────────────────────────────────
print(f"\n{'='*70}")
print("ANALYSIS: Crossover vs Monotonic dataset properties")
print(f"{'='*70}")

cross_rows = [r for r in rows if r['crossover_type'] == 'Crossover']
mono_rows  = [r for r in rows if r['crossover_type'] == 'Monotonic']
near_rows  = [r for r in rows if r['crossover_type'] == 'Near-crossover']

def safe_mean(lst, key):
    vals = []
    for r in lst:
        v = r[key]
        try:
            vals.append(float(v))
        except (ValueError, TypeError):
            pass
    return np.mean(vals) if vals else float('nan')

NUMERIC_PROPS = [
    'avg_clustering', 'transitivity', 'avg_path_length',
    'spectral_gap', 'lambda_2_fiedler', 'modularity',
    'label_assortativity', 'homophily', 'avg_degree',
]

print(f"\n  {'Property':25s}  {'Crossover(3)':>14}  {'Monotonic(4)':>14}  {'Near-cross(2)':>14}  {'Diff%':>8}")
print("  " + "-"*80)
for prop in NUMERIC_PROPS:
    cm = safe_mean(cross_rows, prop)
    mm = safe_mean(mono_rows, prop)
    nm = safe_mean(near_rows, prop)
    diff_pct = ((cm - mm) / (abs(mm) + 1e-9)) * 100 if not np.isnan(cm + mm) else float('nan')
    print(f"  {prop:25s}  {cm:>14.4f}  {mm:>14.4f}  {nm:>14.4f}  {diff_pct:>+7.1f}%")

# Per-dataset summary table
print(f"\n{'='*70}")
print("PER-DATASET SUMMARY")
print(f"{'='*70}")
print(f"  {'Dataset':25s}  {'Type':14}  {'clust':>7}  {'mod':>7}  {'assort':>7}  {'homoph':>7}  {'Fiedler':>9}  {'PartA@10':>9}")
print("  " + "-"*95)
for r in rows:
    def fv(k):
        v = r[k]
        try: return f"{float(v):.4f}"
        except: return "  N/A "
    print(f"  {r['dataset']:25s}  {r['crossover_type']:14}  "
          f"{fv('avg_clustering'):>7}  {fv('modularity'):>7}  "
          f"{fv('label_assortativity'):>7}  {fv('homophily'):>7}  "
          f"{fv('lambda_2_fiedler'):>9}  {r['part_a_k10']:>+9.2f}pp")

# ─── PLOT 1: Bar chart comparison Crossover vs Monotonic ─────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

PROPS_TO_PLOT = [
    ('avg_clustering',      'Avg Clustering Coeff'),
    ('transitivity',        'Transitivity (Global Clust.)'),
    ('modularity',          'Modularity (Label Partition)'),
    ('label_assortativity', 'Label Assortativity'),
    ('homophily',           'Edge Homophily'),
    ('lambda_2_fiedler',    'Fiedler Value λ₂'),
    ('spectral_gap',        'Spectral Gap (λ₂ - λ₁)'),
    ('avg_path_length',     'Avg Path Length'),
]

DS_COLORS = {
    "Crossover":     "#e03030",
    "Near-crossover":"#e07020",
    "Monotonic":     "#2060c0",
}

for ax_i, (prop, label) in enumerate(PROPS_TO_PLOT):
    ax = axes[ax_i]
    ds_names = [r['dataset'] for r in rows]
    vals     = []
    colors   = []
    for r in rows:
        try:    vals.append(float(r[prop]))
        except: vals.append(0.0)
        colors.append(DS_COLORS[r['crossover_type']])

    bars = ax.bar(range(len(ds_names)), vals, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(ds_names)))
    ax.set_xticklabels([d.replace('coauthor-', 'co-').replace('amazon-', 'amz-')
                        for d in ds_names], rotation=45, ha='right', fontsize=7)
    ax.set_title(label, fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=DS_COLORS[t], label=t)
                   for t in ["Crossover", "Near-crossover", "Monotonic"]]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Q3: Graph Topology Properties by Dataset\n"
             "Red=Crossover, Orange=Near-crossover, Blue=Monotonic", fontsize=11)
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
for ext in ['pdf', 'png']:
    p = os.path.join(OUTPUT_DIR, f'plot_q3_crossover_vs_monotonic.{ext}')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved: {p}")
plt.close()

# ─── PLOT 2: Scatter — property vs Part A @ k=10 ─────────────────────────────
fig2, axes2 = plt.subplots(2, 4, figsize=(16, 7))
axes2 = axes2.flatten()

part_a_vals = [r['part_a_k10'] for r in rows]

for ax_i, (prop, label) in enumerate(PROPS_TO_PLOT):
    ax = axes2[ax_i]
    for r in rows:
        try:    pv = float(r[prop])
        except: continue
        pa = r['part_a_k10']
        color = DS_COLORS[r['crossover_type']]
        ax.scatter(pv, pa, color=color, s=60, zorder=3, edgecolors='white', linewidth=0.5)
        ax.annotate(r['dataset'].split('-')[0][:4],
                    xy=(pv, pa), fontsize=6, ha='center', va='bottom',
                    xytext=(0, 3), textcoords='offset points', color=color)

    # Regression line
    prop_vals = []
    pa_vals_reg = []
    for r in rows:
        try:
            prop_vals.append(float(r[prop]))
            pa_vals_reg.append(r['part_a_k10'])
        except: pass
    if len(prop_vals) > 2:
        r_val, p_val = __import__('scipy.stats', fromlist=['pearsonr']).pearsonr(prop_vals, pa_vals_reg)
        x_line = np.linspace(min(prop_vals), max(prop_vals), 50)
        m, b = np.polyfit(prop_vals, pa_vals_reg, 1)
        ax.plot(x_line, m*x_line + b, 'k--', linewidth=1, alpha=0.5)
        ax.set_title(f"{label}\nr={r_val:.2f} p={p_val:.3f}", fontsize=8)
    else:
        ax.set_title(label, fontsize=8)

    ax.axhline(0, color='black', lw=0.7, ls='--', alpha=0.4)
    ax.set_xlabel(prop.replace('_', ' '), fontsize=7)
    ax.set_ylabel("Part A @ k=10 [pp]", fontsize=7)
    ax.grid(True, alpha=0.25)

fig2.suptitle("Q3: Graph Properties vs Part A @ k=10\n"
              "Red=Crossover, Orange=Near-crossover, Blue=Monotonic", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ['pdf', 'png']:
    p = os.path.join(OUTPUT_DIR, f'plot_q3_property_vs_parta.{ext}')
    fig2.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved: {p}")
plt.close()

# ─── Save summary ─────────────────────────────────────────────────────────────
summary_path = os.path.join(OUTPUT_DIR, "q3_summary.txt")
with open(summary_path, "w") as f:
    f.write("Q3: GRAPH TOPOLOGY FEATURES\n" + "="*60 + "\n\n")
    f.write(f"Crossover datasets (3): {[r['dataset'] for r in cross_rows]}\n")
    f.write(f"Near-crossover (2):     {[r['dataset'] for r in near_rows]}\n")
    f.write(f"Monotonic (4):          {[r['dataset'] for r in mono_rows]}\n\n")
    for prop in NUMERIC_PROPS:
        cm = safe_mean(cross_rows, prop)
        mm = safe_mean(mono_rows, prop)
        diff = ((cm - mm) / (abs(mm) + 1e-9)) * 100
        f.write(f"{prop:25s}: Crossover={cm:.4f}  Monotonic={mm:.4f}  diff={diff:+.1f}%\n")
print(f"Saved: {summary_path}")

print(f"\nTotal runtime: {(time.time()-total_start)/60:.1f} min")
print(f"Done. All outputs in: {OUTPUT_DIR}")
