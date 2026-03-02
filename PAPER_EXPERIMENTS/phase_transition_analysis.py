"""
Phase Transition Analysis for Restricted Eigenvector Experiments
================================================================
Extracts and plots:
  1. Eigenvalue distribution stats (amazon-computers, citeseer) at k=[1,10,30]
  2. Condition number evolution (ALL 9 datasets) across all available k
  3. Part A evolution (ALL 9 datasets) across all available k

Sources:
  - analytics/exp2_spectral_analysis_k{k}.json  -> eigenvalue_stats, condition_number_U, ortho_error
  - k{k}/metrics/results.json                   -> framework_analysis (part_a_pp, sgc_mlp, restricted_std)

Outputs (saved to PAPER_EXPERIMENTS/PAPER_OUTPUT/phase_transition/):
  - eigenvalue_stats.csv
  - condition_numbers.csv
  - part_a_evolution.csv
  - plot_condition_number_vs_k.pdf + .png
  - plot_part_a_vs_k.pdf + .png
  - verification_report.txt
"""

import json
import os
import glob
import csv
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "PAPER_RESULTS")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "PAPER_OUTPUT", "phase_transition")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Dataset configuration ────────────────────────────────────────────────────
ALL_9_DATASETS = [
    "amazon-computers",
    "amazon-photo",
    "citeseer",
    "coauthor-cs",
    "coauthor-physics",
    "cora",
    "ogbn-arxiv",
    "pubmed",
    "wikics",
]

# Datasets requested for eigenvalue detail
EIGENVALUE_DETAIL_DATASETS = ["amazon-computers", "citeseer"]

# Datasets known to show crossover behavior (Part A flips sign at some k)
CROSSOVER_DATASETS = {"amazon-computers", "amazon-photo", "wikics"}

# k values requested vs actually available
REQUESTED_K = [1, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
AVAILABLE_K = [1, 2, 4, 6, 8, 10, 12, 20, 30]   # will be confirmed per dataset below

# Split type to use for primary analysis
SPLIT_TYPE = "fixed"

# ─── Helper: locate result directory ─────────────────────────────────────────
def dataset_dir(dataset, split_type="fixed"):
    return os.path.join(RESULTS_DIR, f"{dataset}_{split_type}_lcc")

def analytics_path(dataset, k, split_type="fixed"):
    return os.path.join(dataset_dir(dataset, split_type),
                        "analytics", f"exp2_spectral_analysis_k{k}.json")

def metrics_path(dataset, k, split_type="fixed"):
    return os.path.join(dataset_dir(dataset, split_type),
                        f"k{k}", "metrics", "results.json")

# ─── 1. Discover which k values actually exist ────────────────────────────────
def discover_available_k(dataset, split_type="fixed"):
    pattern = os.path.join(dataset_dir(dataset, split_type),
                           "analytics", "exp2_spectral_analysis_k*.json")
    paths = glob.glob(pattern)
    ks = []
    for p in paths:
        fname = os.path.basename(p)          # exp2_spectral_analysis_k30.json
        k_str = fname.replace("exp2_spectral_analysis_k", "").replace(".json", "")
        try:
            ks.append(int(k_str))
        except ValueError:
            pass
    return sorted(ks)

# ─── 2. Load analytics JSON ───────────────────────────────────────────────────
def load_analytics(dataset, k, split_type="fixed"):
    p = analytics_path(dataset, k, split_type)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

# ─── 3. Load metrics/results.json ────────────────────────────────────────────
def load_metrics(dataset, k, split_type="fixed"):
    p = metrics_path(dataset, k, split_type)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

# ─── Verification helpers ─────────────────────────────────────────────────────
verification_log = []

def check_ortho(dataset, k, data):
    err = data.get("ortho_error")
    if err is None:
        verification_log.append(f"WARN  [{dataset} k={k}] ortho_error not found")
        return
    if err >= 1e-6:
        verification_log.append(
            f"FAIL  [{dataset} k={k}] ortho_error={err:.2e} EXCEEDS 1e-6 threshold!"
        )
    else:
        verification_log.append(
            f"OK    [{dataset} k={k}] ortho_error={err:.2e} < 1e-6"
        )

def check_eigenvalue_range(dataset, k, eig_stats):
    emin = eig_stats.get("min")
    emax = eig_stats.get("max")
    if emin is None or emax is None:
        return
    if emin < -0.01 or emax > 2.1:
        verification_log.append(
            f"FLAG  [{dataset} k={k}] EIGENVALUE OUT OF [0,2]: min={emin:.4f} max={emax:.4f}"
            f" — possible L/D swap!"
        )
    else:
        verification_log.append(
            f"OK    [{dataset} k={k}] eigenvalue range [{emin:.4f}, {emax:.4f}] ⊂ [0,2]"
        )

# ─── TASK 1: Eigenvalue distribution evolution ───────────────────────────────
print("=" * 70)
print("TASK 1: Eigenvalue Distribution Evolution")
print("=" * 70)

EIGENVALUE_K = [1, 10, 30]
eigenvalue_rows = []

for dataset in EIGENVALUE_DETAIL_DATASETS:
    for k in EIGENVALUE_K:
        data = load_analytics(dataset, k, SPLIT_TYPE)
        if data is None:
            print(f"  MISSING: {dataset} k={k} analytics not found")
            continue

        check_ortho(dataset, k, data)
        eig_stats = data["analysis"].get("eigenvalue_stats", {})
        check_eigenvalue_range(dataset, k, eig_stats)

        emin   = eig_stats.get("min")
        emax   = eig_stats.get("max")
        emean  = eig_stats.get("mean")
        estd   = eig_stats.get("std")
        d_eff  = eig_stats.get("d_effective", data.get("d_effective"))

        # Condition number from eigenvalue ratio
        kappa_eig = (emax / emin) if (emin and emin > 0) else float("nan")

        row = dict(
            dataset       = dataset,
            k             = k,
            eig_min       = emin,
            eig_max       = emax,
            eig_mean      = emean,
            eig_std       = estd,
            d_eff         = d_eff,
            kappa_eig     = kappa_eig,       # lambda_max / lambda_min
            kappa_U       = data["analysis"].get("condition_number_U"),   # sigma_max/sigma_min of U
            ortho_error   = data.get("ortho_error"),
        )
        eigenvalue_rows.append(row)
        print(f"  {dataset:25s} k={k:>2d}: "
              f"eig=[{emin:.4f}, {emax:.4f}]  mean={emean:.4f}  std={estd:.4f}  "
              f"d_eff={d_eff}  κ(λ)={kappa_eig:.2f}  κ(U)={row['kappa_U']:.3f}")

# Write CSV
eig_csv = os.path.join(OUTPUT_DIR, "eigenvalue_stats.csv")
with open(eig_csv, "w", newline="") as f:
    fieldnames = ["dataset", "k", "eig_min", "eig_max", "eig_mean", "eig_std",
                  "d_eff", "kappa_eig", "kappa_U", "ortho_error"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(eigenvalue_rows)
print(f"\n  Saved: {eig_csv}")

# ─── TASK 2: Condition Number Evolution (all 9 datasets) ─────────────────────
print()
print("=" * 70)
print("TASK 2: Condition Number Evolution (all 9 datasets)")
print("=" * 70)

missing_k_report = {}   # dataset -> list of missing k values
cond_rows = []

for dataset in ALL_9_DATASETS:
    avail_k = discover_available_k(dataset, SPLIT_TYPE)
    missing = sorted(set(REQUESTED_K) - set(avail_k))
    if missing:
        missing_k_report[dataset] = missing

    print(f"  {dataset:25s}: available k={avail_k}, missing k={missing}")

    for k in avail_k:
        data = load_analytics(dataset, k, SPLIT_TYPE)
        if data is None:
            continue

        check_ortho(dataset, k, data)
        eig_stats = data["analysis"].get("eigenvalue_stats", {})
        check_eigenvalue_range(dataset, k, eig_stats)

        emin   = eig_stats.get("min")
        emax   = eig_stats.get("max")
        kappa_eig = (emax / emin) if (emin and emin > 0) else float("nan")
        kappa_U   = data["analysis"].get("condition_number_U")
        d_eff     = data.get("d_effective")

        cond_rows.append(dict(
            dataset     = dataset,
            k           = k,
            kappa_eig   = kappa_eig,
            kappa_U     = kappa_U,
            eig_min     = emin,
            eig_max     = emax,
            d_eff       = d_eff,
            ortho_error = data.get("ortho_error"),
        ))

cond_csv = os.path.join(OUTPUT_DIR, "condition_numbers.csv")
with open(cond_csv, "w", newline="") as f:
    fieldnames = ["dataset", "k", "kappa_eig", "kappa_U", "eig_min", "eig_max",
                  "d_eff", "ortho_error"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(cond_rows)
print(f"\n  Saved: {cond_csv}")

# ─── TASK 3: Part A Evolution (all 9 datasets) ───────────────────────────────
print()
print("=" * 70)
print("TASK 3: Part A Evolution (all 9 datasets)")
print("=" * 70)

part_a_rows = []

for dataset in ALL_9_DATASETS:
    avail_k = discover_available_k(dataset, SPLIT_TYPE)
    for k in avail_k:
        mdata = load_metrics(dataset, k, SPLIT_TYPE)
        if mdata is None:
            print(f"  MISSING metrics: {dataset} k={k}")
            continue

        fa = mdata.get("framework_analysis", {})
        part_a    = fa.get("part_a_pp")
        sgc_acc   = fa.get("sgc_mlp_acc_pct")
        rest_acc  = fa.get("restricted_std_acc_pct")
        part_b    = fa.get("part_b_pp")
        gap       = fa.get("remaining_gap_pp")

        if part_a is None:
            print(f"  MISSING framework_analysis: {dataset} k={k}")
            continue

        part_a_rows.append(dict(
            dataset              = dataset,
            k                    = k,
            part_a_pp            = round(part_a, 4),
            sgc_mlp_acc_pct      = round(sgc_acc, 4) if sgc_acc is not None else None,
            restricted_std_acc_pct = round(rest_acc, 4) if rest_acc is not None else None,
            part_b_pp            = round(part_b, 4) if part_b is not None else None,
            remaining_gap_pp     = round(gap, 4) if gap is not None else None,
        ))

    # Print summary for this dataset
    ds_rows = [r for r in part_a_rows if r["dataset"] == dataset]
    if ds_rows:
        vals = [(r["k"], r["part_a_pp"]) for r in ds_rows]
        print(f"  {dataset:25s}: " + "  ".join(f"k={k}:{v:+.1f}pp" for k,v in vals))

part_a_csv = os.path.join(OUTPUT_DIR, "part_a_evolution.csv")
with open(part_a_csv, "w", newline="") as f:
    fieldnames = ["dataset", "k", "part_a_pp", "sgc_mlp_acc_pct",
                  "restricted_std_acc_pct", "part_b_pp", "remaining_gap_pp"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(part_a_rows)
print(f"\n  Saved: {part_a_csv}")

# ─── PLOT 1: Condition Number vs k ───────────────────────────────────────────
print()
print("=" * 70)
print("PLOT 1: Condition Number (κ of U) vs k")
print("=" * 70)

# Use σ-based condition number (kappa_U) — this is the matrix condition number of U
# kappa_eig = λ_max/λ_min is the eigenvalue spread, also shown dashed

COLORS = plt.cm.tab10.colors
DATASET_COLOR = {ds: COLORS[i % 10] for i, ds in enumerate(ALL_9_DATASETS)}

fig, ax = plt.subplots(figsize=(9, 5.5))

for dataset in ALL_9_DATASETS:
    rows = sorted([r for r in cond_rows if r["dataset"] == dataset], key=lambda r: r["k"])
    if not rows:
        continue
    ks     = [r["k"] for r in rows]
    kappas = [r["kappa_U"] for r in rows]
    color  = DATASET_COLOR[dataset]
    lw     = 2.2 if dataset in CROSSOVER_DATASETS else 1.4
    ls     = "-"
    zorder = 3 if dataset in CROSSOVER_DATASETS else 2
    label  = dataset + ("  ✕" if dataset in CROSSOVER_DATASETS else "")

    ax.plot(ks, kappas, color=color, linewidth=lw, linestyle=ls,
            marker="o", markersize=4, label=label, zorder=zorder)

    # Highlight crossover datasets in red outline
    if dataset in CROSSOVER_DATASETS:
        ax.plot(ks, kappas, color="red", linewidth=2.8, linestyle=ls,
                alpha=0.35, zorder=zorder - 1)

ax.set_xlabel("Diffusion depth k", fontsize=12)
ax.set_ylabel("Condition number κ(U)  [σ_max / σ_min]", fontsize=12)
ax.set_title("Condition Number of Restricted Eigenvector Basis U vs k\n"
             "(crossover datasets highlighted in red)", fontsize=12)
ax.set_xticks(AVAILABLE_K)
ax.set_xticklabels([str(k) for k in AVAILABLE_K])
ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.85)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax.grid(True, which="both", alpha=0.3)
ax.annotate("Missing k=15, 25 (not run)", xy=(0.98, 0.02),
            xycoords="axes fraction", ha="right", fontsize=8, color="gray")

plt.tight_layout()
for ext in ["pdf", "png"]:
    p = os.path.join(OUTPUT_DIR, f"plot_condition_number_vs_k.{ext}")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p}")
plt.close()

# ─── PLOT 2: Part A vs k ─────────────────────────────────────────────────────
print()
print("PLOT 2: Part A vs k")
print("=" * 70)

fig, ax = plt.subplots(figsize=(9, 5.5))

# Draw zero line
ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5, zorder=1)

for dataset in ALL_9_DATASETS:
    rows = sorted([r for r in part_a_rows if r["dataset"] == dataset], key=lambda r: r["k"])
    if not rows:
        continue
    ks      = [r["k"] for r in rows]
    part_as = [r["part_a_pp"] for r in rows]
    color   = DATASET_COLOR[dataset]
    lw      = 2.2 if dataset in CROSSOVER_DATASETS else 1.4
    zorder  = 3 if dataset in CROSSOVER_DATASETS else 2
    label   = dataset + ("  ✕" if dataset in CROSSOVER_DATASETS else "")

    ax.plot(ks, part_as, color=color, linewidth=lw, linestyle="-",
            marker="o", markersize=4, label=label, zorder=zorder)

    if dataset in CROSSOVER_DATASETS:
        ax.plot(ks, part_as, color="red", linewidth=2.8, alpha=0.35, zorder=zorder - 1)

        # Mark the crossover point (where Part A crosses 0)
        for i in range(len(part_as) - 1):
            if part_as[i] * part_as[i + 1] < 0:  # sign change
                # Linear interpolation of zero crossing
                k0, k1 = ks[i], ks[i + 1]
                v0, v1 = part_as[i], part_as[i + 1]
                k_cross = k0 + (0 - v0) * (k1 - k0) / (v1 - v0)
                ax.axvline(k_cross, color=color, linewidth=1.0, linestyle=":",
                           alpha=0.7, zorder=1)
                ax.annotate(f"↕ {dataset}\nk≈{k_cross:.1f}",
                            xy=(k_cross, 0), xytext=(k_cross + 0.5, 8),
                            fontsize=6.5, color=color,
                            arrowprops=dict(arrowstyle="-", color=color, lw=0.8))
                break

ax.set_xlabel("Diffusion depth k", fontsize=12)
ax.set_ylabel("Part A  [pp]  (SGC+MLP − Restricted+Std)", fontsize=12)
ax.set_title("Part A Evolution vs k  (all 9 datasets, fixed splits)\n"
             "Part A > 0: SGC beats U (eigenvector-detrimental)   "
             "Part A < 0: U beats SGC", fontsize=10.5)
ax.set_xticks(AVAILABLE_K)
ax.set_xticklabels([str(k) for k in AVAILABLE_K])
ax.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.85)
ax.grid(True, alpha=0.3)
ax.annotate("Missing k=15, 25 (not run)", xy=(0.98, 0.02),
            xycoords="axes fraction", ha="right", fontsize=8, color="gray")

plt.tight_layout()
for ext in ["pdf", "png"]:
    p = os.path.join(OUTPUT_DIR, f"plot_part_a_vs_k.{ext}")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p}")
plt.close()

# ─── Verification Report ─────────────────────────────────────────────────────
print()
print("=" * 70)
print("VERIFICATION REPORT")
print("=" * 70)

# Summarize missing k values
print("\n--- Missing k values (relative to requested [1,2,4,6,8,10,12,15,20,25,30]) ---")
global_missing = sorted(set(REQUESTED_K) - set(AVAILABLE_K))
print(f"  Globally absent: k={global_missing}  (k=15 and k=25 were never run)")
for ds, missing in sorted(missing_k_report.items()):
    dataset_specific = sorted(set(missing) - set(global_missing))
    if dataset_specific:
        print(f"  {ds}: additionally missing k={dataset_specific}")

# Ortho error summary
fails = [l for l in verification_log if "FAIL" in l]
flags = [l for l in verification_log if "FLAG" in l]
oks   = [l for l in verification_log if "OK" in l and "ortho" in l.lower()]
print(f"\n--- D-Orthonormality (U^T D U ≈ I, threshold 1e-6) ---")
print(f"  PASS: {len(oks)} checks")
print(f"  FAIL: {len(fails)} checks")
for l in fails:
    print(f"    {l}")

print(f"\n--- Eigenvalue Range [0, 2] ---")
if flags:
    for l in flags:
        print(f"  {l}")
else:
    print(f"  All eigenvalues within [0, 2] — no L/D swap detected")

# Save full verification log
vlog_path = os.path.join(OUTPUT_DIR, "verification_report.txt")
with open(vlog_path, "w") as f:
    f.write("PHASE TRANSITION ANALYSIS — VERIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write("Missing k values (globally): " + str(global_missing) + "\n\n")
    f.write("D-Orthonormality checks:\n")
    for l in verification_log:
        f.write("  " + l + "\n")
print(f"\n  Full log saved: {vlog_path}")

# ─── Final summary table ─────────────────────────────────────────────────────
print()
print("=" * 70)
print("SUMMARY: Part A at key k values")
print("=" * 70)
print(f"  {'Dataset':25s}  {'k=1':>8}  {'k=4':>8}  {'k=10':>8}  {'k=30':>8}")
print("  " + "-" * 60)
for dataset in ALL_9_DATASETS:
    vals = {r["k"]: r["part_a_pp"] for r in part_a_rows if r["dataset"] == dataset}
    def v(k):
        return f"{vals[k]:+.1f}pp" if k in vals else "  N/A  "
    print(f"  {dataset:25s}  {v(1):>8}  {v(4):>8}  {v(10):>8}  {v(30):>8}")

print()
print(f"  {'Dataset':25s}  {'κ(U) k=1':>10}  {'κ(U) k=10':>10}  {'κ(U) k=30':>10}")
print("  " + "-" * 60)
for dataset in ALL_9_DATASETS:
    vals = {r["k"]: r["kappa_U"] for r in cond_rows if r["dataset"] == dataset}
    def v(k):
        return f"{vals[k]:.3f}" if k in vals else "  N/A  "
    print(f"  {dataset:25s}  {v(1):>10}  {v(10):>10}  {v(30):>10}")

print()
print("Done. All outputs in:", OUTPUT_DIR)
