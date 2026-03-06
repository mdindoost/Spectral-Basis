"""
LR Convergence Audit
====================
Reruns the logistic regression (p=all, p=50) experiment at k=10, fixed split,
all 9 datasets with explicit convergence-safe settings:
  C=1.0, max_iter=10000, solver='lbfgs', multi_class='multinomial', tol=1e-4

Reports per dataset:
  - LR accuracy at p=all and p=50
  - Whether sklearn raised a ConvergenceWarning
  - clf.n_iter_ (actual iterations used)
  - Comparison against previously reported values (old max_iter=1000 results)

Old reference values (from q1_label_alignment.py output, max_iter=1000):
  cora              39.2%
  citeseer          31.4%
  pubmed            47.9%
  ogbn-arxiv        67.1%
  wikics            64.9%
  amazon-computers  88.3%
  amazon-photo      89.1%
  coauthor-cs       68.3%
  coauthor-physics  86.6%
"""

import sys, os, warnings, time
sys.stdout.reconfigure(line_buffering=True)

import torch as _torch
_torch_load_orig = _torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load_orig(*args, **kwargs)
_torch.load = _torch_load_compat

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
DATASET_ROOT = os.path.join(REPO_ROOT, "dataset")

from src.graph_utils import (
    load_dataset, build_graph_matrices,
    get_largest_connected_component_nx, extract_subgraph,
    compute_sgc_normalized_adjacency, sgc_precompute,
    compute_restricted_eigenvectors,
)

ALL_9 = [
    "cora", "citeseer", "pubmed",
    "ogbn-arxiv", "wikics",
    "amazon-computers", "amazon-photo",
    "coauthor-cs", "coauthor-physics",
]
K_TARGET = 10
SPLIT    = "fixed"

# Previously reported p=all values (max_iter=1000, multi_class='auto')
OLD_VALUES = {
    "cora":             39.2,
    "citeseer":         31.4,
    "pubmed":           47.9,
    "ogbn-arxiv":       67.1,
    "wikics":           64.9,
    "amazon-computers": 88.3,
    "amazon-photo":     89.1,
    "coauthor-cs":      68.3,
    "coauthor-physics": 86.6,
}


def load_and_preprocess(dataset_name):
    (edge_index, X_raw, labels, num_nodes, num_classes,
     train_idx_orig, val_idx_orig, test_idx_orig) = load_dataset(dataset_name, root=DATASET_ROOT)

    adj, L, D = build_graph_matrices(edge_index, num_nodes)
    lcc_mask = get_largest_connected_component_nx(adj)
    split_idx_original = {
        'train_idx': train_idx_orig,
        'val_idx':   val_idx_orig,
        'test_idx':  test_idx_orig,
    }
    adj, X_raw, labels, split_idx = extract_subgraph(
        adj, X_raw, labels, lcc_mask, split_idx_original
    )

    adj_no_loops   = adj - sp.diags(adj.diagonal())
    adj_coo        = adj_no_loops.tocoo()
    edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
    adj, L, D      = build_graph_matrices(edge_index_lcc, adj.shape[0])

    features_dense = X_raw.toarray() if sp.issparse(X_raw) else X_raw.copy()
    adj_norm       = compute_sgc_normalized_adjacency(adj)

    return dict(
        features_dense = features_dense,
        labels         = labels,
        num_classes    = len(np.unique(labels)),
        adj_norm       = adj_norm,
        L              = L,
        D              = D,
        train_idx      = split_idx['train_idx'],
        val_idx        = split_idx['val_idx'],
        test_idx       = split_idx['test_idx'],
    )


def run_lr(X_train, y_train, X_test, y_test):
    """Run LR with explicit settings; return (acc_pct, converged, n_iter)."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = LogisticRegression(
        C=1.0,
        max_iter=10000,
        solver='lbfgs',
        multi_class='multinomial',
        tol=1e-4,
        random_state=42,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        clf.fit(X_train, y_train)
        converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)

    acc = float(clf.score(X_test, y_test)) * 100
    n_iter = int(clf.n_iter_.max())   # n_iter_ is array (one per class in OvR; scalar for multinomial)
    return acc, converged, n_iter


# ─── Main ─────────────────────────────────────────────────────────────────────

print("=" * 78)
print("LR CONVERGENCE AUDIT  —  k=10, fixed split, all 9 datasets")
print("Settings: C=1.0, max_iter=10000, solver=lbfgs, multi_class=multinomial, tol=1e-4")
print("=" * 78)

results = []

for dataset_name in ALL_9:
    t0 = time.time()
    print(f"\n[{dataset_name}]  loading...", flush=True)

    ds = load_and_preprocess(dataset_name)

    train_idx = ds['train_idx']
    test_idx  = ds['test_idx']
    labels    = ds['labels']

    # SGC diffusion at k=10
    X_diff = sgc_precompute(ds['features_dense'].copy(), ds['adj_norm'], K_TARGET)

    # Rayleigh-Ritz (identical to master_training.py)
    U, eigenvalues, d_eff, ortho_error = compute_restricted_eigenvectors(
        X_diff, ds['L'], ds['D'], num_components=0
    )

    ortho_ok = ortho_error < 1e-6
    print(f"  d_eff={d_eff}  ortho_error={ortho_error:.2e}  {'OK' if ortho_ok else 'FAIL'}")

    U_train = U[train_idx]
    U_test  = U[test_idx]
    y_train = labels[train_idx]
    y_test  = labels[test_idx]

    # ── p=all ────────────────────────────────────────────────────────────────
    print(f"  Running LR p=all (d_eff={d_eff})...", flush=True)
    acc_all, conv_all, niter_all = run_lr(U_train, y_train, U_test, y_test)

    # ── p=50 ─────────────────────────────────────────────────────────────────
    p50 = min(50, d_eff)
    print(f"  Running LR p=50 (using {p50} dims)...", flush=True)
    acc_50, conv_50, niter_50 = run_lr(U_train[:, :p50], y_train, U_test[:, :p50], y_test)

    old_val = OLD_VALUES.get(dataset_name)
    delta   = round(acc_all - old_val, 2) if old_val is not None else None
    changed = abs(delta) > 1.0 if delta is not None else None

    results.append(dict(
        dataset      = dataset_name,
        d_eff        = d_eff,
        num_classes  = ds['num_classes'],
        # p=all
        acc_all      = round(acc_all, 2),
        conv_all     = conv_all,
        niter_all    = niter_all,
        # p=50
        acc_50       = round(acc_50, 2),
        conv_50      = conv_50,
        niter_50     = niter_50,
        # comparison
        old_acc_all  = old_val,
        delta_pp     = delta,
        changed_flag = changed,
        ortho_ok     = ortho_ok,
    ))

    elapsed = time.time() - t0
    print(f"  p=all: {acc_all:.2f}%  converged={conv_all}  n_iter={niter_all}")
    print(f"  p=50:  {acc_50:.2f}%  converged={conv_50}  n_iter={niter_50}")
    print(f"  Old p=all: {old_val}%  delta={delta:+.2f}pp  {'*** CHANGED >1pp ***' if changed else ''}")
    print(f"  Elapsed: {elapsed:.1f}s")


# ─── Summary table ────────────────────────────────────────────────────────────
print()
print("=" * 78)
print("SUMMARY TABLE")
print("=" * 78)
print(f"  {'Dataset':20s}  {'d_eff':>6}  {'C':>2}  "
      f"{'p=all':>7}  {'conv':>5}  {'niter':>6}  "
      f"{'p=50':>7}  {'conv':>5}  {'niter':>6}  "
      f"{'old':>7}  {'delta':>7}  {'flag'}")
print("  " + "-" * 102)

flags = []
for r in results:
    conv_sym_all = "YES" if r['conv_all'] else "NO "
    conv_sym_50  = "YES" if r['conv_50']  else "NO "

    near_max_all = r['niter_all'] >= 9000
    near_max_50  = r['niter_50']  >= 9000

    flag_parts = []
    if not r['conv_all']:
        flag_parts.append("DID-NOT-CONVERGE(p=all)")
    if not r['conv_50']:
        flag_parts.append("DID-NOT-CONVERGE(p=50)")
    if near_max_all:
        flag_parts.append(f"NEAR-MAX-ITER(p=all,n={r['niter_all']})")
    if near_max_50:
        flag_parts.append(f"NEAR-MAX-ITER(p=50,n={r['niter_50']})")
    if r['changed_flag']:
        flag_parts.append(f"CHANGED({r['delta_pp']:+.2f}pp)")

    flag_str = " | ".join(flag_parts) if flag_parts else "-"
    if flag_parts:
        flags.append((r['dataset'], flag_str))

    print(f"  {r['dataset']:20s}  {r['d_eff']:>6}  {r['num_classes']:>2}  "
          f"  {r['acc_all']:>5.1f}%  {conv_sym_all}  {r['niter_all']:>6}  "
          f"  {r['acc_50']:>5.1f}%  {conv_sym_50}  {r['niter_50']:>6}  "
          f"  {str(r['old_acc_all'])+'%':>7}  {('+' if r['delta_pp'] and r['delta_pp']>0 else '')+str(r['delta_pp'])+'pp':>8}  "
          f"{flag_str}")

print()
print("=" * 78)
print("FLAGS REQUIRING ATTENTION")
print("=" * 78)
if flags:
    for ds, f in flags:
        print(f"  *** {ds}: {f}")
else:
    print("  None — all datasets converged and no values changed >1pp.")

print()
print("=" * 78)
print("COAUTHOR-CS FOCUS (key exception: MLP +9pp over LR)")
print("=" * 78)
cs = next((r for r in results if r['dataset'] == 'coauthor-cs'), None)
if cs:
    print(f"  d_eff={cs['d_eff']}  num_classes={cs['num_classes']}")
    print(f"  LR p=all (new): {cs['acc_all']:.2f}%  converged={cs['conv_all']}  n_iter={cs['niter_all']}")
    print(f"  LR p=all (old): {cs['old_acc_all']}%  delta={cs['delta_pp']:+.2f}pp")
    if cs['delta_pp'] and cs['delta_pp'] > 1.0:
        print("  *** LR improved >1pp with proper convergence — MLP exception weakens ***")
    elif cs['delta_pp'] and cs['delta_pp'] <= 1.0 and cs['conv_all']:
        print("  LR converged and result stable — MLP exception stands.")
    elif not cs['conv_all']:
        print("  *** STILL DID NOT CONVERGE at max_iter=10000 — problem is genuinely hard ***")
