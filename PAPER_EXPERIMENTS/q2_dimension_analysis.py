"""
Q2: Dimension Loss vs Crossover Analysis
=========================================
All data comes from existing JSON files — no new experiments needed.

Sources:
  analytics/exp2_spectral_analysis_k*.json  →  d_effective
  k{k}/metrics/results.json                 →  framework_analysis.part_a_pp

Outputs (PAPER_OUTPUT/q2_analysis/):
  q2_d_eff_evolution.csv
  q2_dimension_correlation.csv
  q2_crossover_dimensions.csv
  plot_q2_dual_axis.pdf/.png   (d_eff retention + Part A vs k, selected datasets)
  plot_q2_all_datasets.pdf/.png (all 9 datasets, 3×3 panel)
  q2_summary.txt
"""

import json, os, glob, csv, warnings
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE, "PAPER_RESULTS")
OUTPUT_DIR  = os.path.join(BASE, "PAPER_OUTPUT", "q2_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_9 = [
    "amazon-computers", "amazon-photo", "citeseer",
    "coauthor-cs", "coauthor-physics", "cora",
    "ogbn-arxiv", "pubmed", "wikics",
]
SPLIT = "fixed"

# From prior analysis — datasets where Part A crosses zero
CROSSOVER_DATASETS = {"amazon-computers", "amazon-photo", "ogbn-arxiv"}
# Wikics approaching zero at k=30 but hasn't crossed
NEAR_CROSSOVER     = {"wikics"}

AVAILABLE_K = [1, 2, 4, 6, 8, 10, 12, 20, 30]

# ─── Loaders ─────────────────────────────────────────────────────────────────
def load_analytics(dataset, k):
    p = os.path.join(RESULTS_DIR, f"{dataset}_{SPLIT}_lcc",
                     "analytics", f"exp2_spectral_analysis_k{k}.json")
    return json.load(open(p)) if os.path.exists(p) else None

def load_metrics(dataset, k):
    p = os.path.join(RESULTS_DIR, f"{dataset}_{SPLIT}_lcc",
                     f"k{k}", "metrics", "results.json")
    return json.load(open(p)) if os.path.exists(p) else None

# ─── STEP 1: Collect d_eff and Part A series per dataset ─────────────────────
print("=" * 70)
print("STEP 1: Collecting d_eff and Part A series")
print("=" * 70)

series = {}   # dataset -> {k: {d_eff, part_a, sgc_acc, rest_acc}}

for ds in ALL_9:
    series[ds] = {}
    for k in AVAILABLE_K:
        an = load_analytics(ds, k)
        me = load_metrics(ds, k)
        if an is None or me is None:
            continue
        fa = me.get("framework_analysis", {})
        part_a = fa.get("part_a_pp")
        if part_a is None:
            continue
        series[ds][k] = dict(
            k             = k,
            d_eff         = an.get("d_effective"),
            part_a_pp     = round(part_a, 4),
            sgc_mlp_acc   = round(fa.get("sgc_mlp_acc_pct", float("nan")), 4),
            rest_std_acc  = round(fa.get("restricted_std_acc_pct", float("nan")), 4),
            ortho_error   = an.get("ortho_error"),
        )
    ks_found = sorted(series[ds].keys())
    print(f"  {ds:25s}: {len(ks_found)} k-values found: {ks_found}")

# ─── STEP 2: Compute d_eff retention rates ───────────────────────────────────
print()
print("=" * 70)
print("STEP 2: d_eff retention rates (d_eff(k) / d_eff(k=1))")
print("=" * 70)

evo_rows = []

for ds in ALL_9:
    d1 = series[ds].get(1, {}).get("d_eff")
    if d1 is None or d1 == 0:
        print(f"  {ds:25s}: WARNING — d_eff at k=1 missing or zero, skipping")
        continue

    print(f"\n  {ds} (d_eff at k=1 = {d1})")
    print(f"  {'k':>4}  {'d_eff':>8}  {'retention':>10}  {'Part A':>9}")

    for k in AVAILABLE_K:
        entry = series[ds].get(k)
        if entry is None:
            continue
        d_k   = entry["d_eff"]
        ret   = d_k / d1
        pa    = entry["part_a_pp"]
        print(f"  {k:>4}  {d_k:>8}  {ret:>10.4f}  {pa:>+9.2f}pp")

        evo_rows.append(dict(
            dataset              = ds,
            k                    = k,
            d_eff                = d_k,
            d_eff_k1             = d1,
            d_eff_retention_rate = round(ret, 6),
            part_a_pp            = pa,
            sgc_mlp_acc          = entry["sgc_mlp_acc"],
            rest_std_acc         = entry["rest_std_acc"],
            ortho_error          = entry["ortho_error"],
        ))

# Write CSV 1
evo_csv = os.path.join(OUTPUT_DIR, "q2_d_eff_evolution.csv")
with open(evo_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "dataset", "k", "d_eff", "d_eff_k1", "d_eff_retention_rate",
        "part_a_pp", "sgc_mlp_acc", "rest_std_acc", "ortho_error"
    ])
    writer.writeheader()
    writer.writerows(evo_rows)
print(f"\n  Saved: {evo_csv}")

# ─── STEP 3: Correlation d_eff_retention vs Part A ───────────────────────────
print()
print("=" * 70)
print("STEP 3: Correlation(d_eff_retention_rate, part_a_pp) per dataset")
print("=" * 70)

corr_rows = []
CORR_THRESHOLD = 0.35   # |r| below this → Neutral

print(f"\n  {'Dataset':25s}  {'r':>7}  {'p-value':>9}  {'n':>4}  {'Pattern'}")
print("  " + "-" * 65)

for ds in ALL_9:
    rows = [r for r in evo_rows if r["dataset"] == ds]
    if len(rows) < 4:
        print(f"  {ds:25s}: insufficient data ({len(rows)} points)")
        continue
    rets = [r["d_eff_retention_rate"] for r in rows]
    pas  = [r["part_a_pp"] for r in rows]
    ks   = [r["k"] for r in rows]

    # If d_eff is perfectly constant, Pearson is undefined — flag separately
    if len(set(rets)) == 1:
        r, p = float("nan"), float("nan")
        pattern = "Constant d_eff"
        stars = ""
        print(f"  {ds:25s}  {'NaN':>7}  {'NaN':>9}     {len(rows):>4}  {pattern}  "
              f"(d_eff={rows[0]['d_eff']} unchanged k=1..{ks[-1]})")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, p = stats.pearsonr(rets, pas)
        # Classify
        if   r < -CORR_THRESHOLD: pattern = "Protective"   # losing dims → higher Part A
        elif r >  CORR_THRESHOLD: pattern = "Harmful"      # losing dims → lower Part A
        else:                     pattern = "Neutral"
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {ds:25s}  {r:>+7.3f}  {p:>9.4f}{stars:<4}  {len(rows):>4}  {pattern}")

    ret_at_k30 = next((r2["d_eff_retention_rate"] for r2 in rows if r2["k"] == 30), float("nan"))
    corr_rows.append(dict(
        dataset                = ds,
        pearson_r              = round(r, 5) if not np.isnan(r) else "NaN",
        p_value                = round(p, 6) if not np.isnan(p) else "NaN",
        n_k_values             = len(rows),
        pattern                = pattern,
        crossover_type         = ("Crossover" if ds in CROSSOVER_DATASETS
                                  else ("Near-crossover" if ds in NEAR_CROSSOVER
                                        else "Monotonic")),
        d_eff_range            = f"{min(r2['d_eff'] for r2 in rows)}–{max(r2['d_eff'] for r2 in rows)}",
        d_eff_retention_k30    = round(ret_at_k30, 4),
        part_a_at_k30          = next((r2["part_a_pp"] for r2 in rows if r2["k"] == 30), None),
    ))

corr_csv = os.path.join(OUTPUT_DIR, "q2_dimension_correlation.csv")
with open(corr_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "dataset", "pearson_r", "p_value", "n_k_values", "pattern",
        "crossover_type", "d_eff_range", "d_eff_retention_k30", "part_a_at_k30"
    ])
    writer.writeheader()
    writer.writerows(corr_rows)
print(f"\n  Saved: {corr_csv}")

# ─── STEP 4: Crossover detection ─────────────────────────────────────────────
print()
print("=" * 70)
print("STEP 4: Crossover detection (Part A sign change)")
print("=" * 70)

crossover_rows = []

for ds in ALL_9:
    rows = sorted([r for r in evo_rows if r["dataset"] == ds], key=lambda r: r["k"])
    if not rows:
        continue

    ks  = [r["k"] for r in rows]
    pas = [r["part_a_pp"] for r in rows]
    d1  = rows[0]["d_eff_k1"]

    # Find sign change
    k_cross = None
    d_at_cross = None
    ret_at_cross = None
    for i in range(len(pas) - 1):
        if pas[i] * pas[i+1] < 0:
            # Linear interpolation
            k0, k1 = ks[i], ks[i+1]
            v0, v1 = pas[i], pas[i+1]
            k_cross = k0 + (0 - v0) * (k1 - k0) / (v1 - v0)
            # Interpolate d_eff at crossover
            d0 = rows[i]["d_eff"]
            d1k = rows[i+1]["d_eff"]
            frac = (k_cross - k0) / (k1 - k0)
            d_at_cross = d0 + frac * (d1k - d0)
            ret_at_cross = d_at_cross / d1
            break

    # Minimum Part A (most negative)
    min_pa = min(pas)
    min_pa_k = ks[pas.index(min_pa)]

    # Is Part A monotonically decreasing?
    diffs = [pas[i+1] - pas[i] for i in range(len(pas)-1)]
    is_monotone_dec = all(d <= 0.5 for d in diffs)   # allow 0.5pp noise

    if k_cross is not None:
        print(f"  {ds:25s}: CROSSOVER at k≈{k_cross:.1f}  "
              f"d_eff≈{d_at_cross:.0f}  retention≈{ret_at_cross:.3f}")
    elif min_pa < 3:
        print(f"  {ds:25s}: NEAR-ZERO  min Part A={min_pa:.1f}pp at k={min_pa_k}  "
              f"(no sign change observed)")
    else:
        print(f"  {ds:25s}: MONOTONIC  Part A range [{min_pa:.1f}, {max(pas):.1f}]pp  "
              f"{'(decreasing)' if is_monotone_dec else '(non-monotone)'}")

    crossover_rows.append(dict(
        dataset                    = ds,
        crossover_type             = ("Crossover" if k_cross is not None
                                      else ("Near-crossover" if ds in NEAR_CROSSOVER
                                            else "Monotonic")),
        k_crossover                = round(k_cross, 2) if k_cross is not None else "N/A",
        d_eff_at_crossover         = round(d_at_cross, 1) if d_at_cross is not None else "N/A",
        retention_at_crossover     = round(ret_at_cross, 4) if ret_at_cross is not None else "N/A",
        part_a_at_k1               = pas[0],
        part_a_at_k10              = next((r["part_a_pp"] for r in rows if r["k"]==10), None),
        part_a_at_k30              = pas[-1] if rows[-1]["k"]==30 else None,
        part_a_min                 = round(min_pa, 2),
        part_a_min_at_k            = min_pa_k,
        d_eff_k1                   = d1,
        d_eff_k30                  = rows[-1]["d_eff"] if rows[-1]["k"]==30 else None,
        d_eff_retention_k30        = round(rows[-1]["d_eff_retention_rate"], 4) if rows[-1]["k"]==30 else None,
        is_monotone_decreasing     = is_monotone_dec,
    ))

cross_csv = os.path.join(OUTPUT_DIR, "q2_crossover_dimensions.csv")
with open(cross_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "dataset", "crossover_type", "k_crossover",
        "d_eff_at_crossover", "retention_at_crossover",
        "part_a_at_k1", "part_a_at_k10", "part_a_at_k30",
        "part_a_min", "part_a_min_at_k",
        "d_eff_k1", "d_eff_k30", "d_eff_retention_k30",
        "is_monotone_decreasing",
    ])
    writer.writeheader()
    writer.writerows(crossover_rows)
print(f"\n  Saved: {cross_csv}")

# ─── STEP 5: Plot A — Dual-axis (selected datasets) ──────────────────────────
print()
print("=" * 70)
print("PLOT A: Dual-axis d_eff retention + Part A vs k (selected datasets)")
print("=" * 70)

# Select: 3 crossover + 2 monotonic reference datasets
SELECTED = [
    ("amazon-computers", "Crossover",     "#e03030"),
    ("amazon-photo",     "Crossover",     "#e07020"),
    ("ogbn-arxiv",       "Crossover",     "#c040c0"),
    ("wikics",           "Near-crossover","#208050"),
    ("cora",             "Monotonic",     "#2060c0"),
    ("citeseer",         "Monotonic",     "#808080"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for ax_i, (ds, dtype, color) in enumerate(SELECTED):
    ax = axes[ax_i]
    rows = sorted([r for r in evo_rows if r["dataset"] == ds], key=lambda r: r["k"])
    if not rows:
        ax.set_visible(False)
        continue

    ks   = [r["k"] for r in rows]
    rets = [r["d_eff_retention_rate"] * 100 for r in rows]   # in %
    pas  = [r["part_a_pp"] for r in rows]

    ax2 = ax.twinx()

    # Left axis: d_eff retention (%)
    l1, = ax.plot(ks, rets, color="#4488cc", linewidth=2, marker="s",
                  markersize=5, label="d_eff retention %")
    ax.fill_between(ks, rets, alpha=0.12, color="#4488cc")
    ax.set_ylabel("d_eff retention  (%)", color="#4488cc", fontsize=9)
    ax.tick_params(axis="y", labelcolor="#4488cc")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

    # Right axis: Part A (pp)
    l2, = ax2.plot(ks, pas, color=color, linewidth=2.2, marker="o",
                   markersize=5, label="Part A (pp)")
    ax2.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.5)
    ax2.set_ylabel("Part A  [pp]", color=color, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=color)

    # Mark crossover if present
    for i in range(len(pas) - 1):
        if pas[i] * pas[i+1] < 0:
            k0, k1 = ks[i], ks[i+1]
            v0, v1 = pas[i], pas[i+1]
            k_c = k0 + (0 - v0) * (k1 - k0) / (v1 - v0)
            ax2.axvline(k_c, color=color, linewidth=1.2, linestyle=":",
                        alpha=0.8, zorder=1)
            ax2.annotate(f"k≈{k_c:.1f}", xy=(k_c, 0),
                         xytext=(k_c + 0.8, max(pas) * 0.35),
                         fontsize=7.5, color=color,
                         arrowprops=dict(arrowstyle="-", color=color, lw=0.7))
            break

    ax.set_xlabel("Diffusion depth k", fontsize=9)
    ax.set_xticks([1, 4, 8, 12, 20, 30])
    ax.set_title(f"{ds}\n[{dtype}]", fontsize=10, color=color, fontweight="bold")
    ax.grid(True, alpha=0.25)

    # Unified legend
    ax.legend(handles=[l1, l2], fontsize=7.5, loc="upper right", framealpha=0.75)

fig.suptitle(
    "Q2: d_eff Retention Rate vs Part A  (fixed splits)\n"
    "Left axis: fraction of initial eigenvector dimensions retained  "
    "Right axis: Part A gap [pp]",
    fontsize=11
)
plt.tight_layout(rect=[0, 0, 1, 0.95])

for ext in ["pdf", "png"]:
    p = os.path.join(OUTPUT_DIR, f"plot_q2_dual_axis.{ext}")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p}")
plt.close()

# ─── STEP 6: Plot B — All 9 datasets in 3×3 panel ────────────────────────────
print()
print("PLOT B: All 9 datasets (3×3 panel)")

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

fig2, axes2 = plt.subplots(3, 3, figsize=(14, 10))
axes2 = axes2.flatten()

for ax_i, ds in enumerate(ALL_9):
    ax = axes2[ax_i]
    rows = sorted([r for r in evo_rows if r["dataset"] == ds], key=lambda r: r["k"])
    if not rows:
        ax.set_visible(False)
        continue

    ks    = [r["k"] for r in rows]
    rets  = [r["d_eff_retention_rate"] * 100 for r in rows]
    pas   = [r["part_a_pp"] for r in rows]
    color = COLORS_9[ds]

    # Categorise
    if ds in CROSSOVER_DATASETS:    cat, ls = "Crossover", "-"
    elif ds in NEAR_CROSSOVER:      cat, ls = "Near-crossover", "--"
    else:                           cat, ls = "Monotonic", "-"

    ax2 = ax.twinx()

    ax.plot(ks, rets, color="#4488cc", linewidth=1.6, marker="s",
            markersize=3.5, linestyle=ls)
    ax.fill_between(ks, rets, alpha=0.10, color="#4488cc")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.tick_params(axis="y", labelcolor="#4488cc", labelsize=7)

    ax2.plot(ks, pas, color=color, linewidth=2, marker="o",
             markersize=3.5, linestyle=ls)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax2.tick_params(axis="y", labelcolor=color, labelsize=7)

    # Mark crossover
    for i in range(len(pas) - 1):
        if pas[i] * pas[i+1] < 0:
            k0, k1 = ks[i], ks[i+1]
            v0, v1 = pas[i], pas[i+1]
            k_c = k0 + (0 - v0) * (k1 - k0) / (v1 - v0)
            ax2.axvline(k_c, color=color, linewidth=1.0, linestyle=":",
                        alpha=0.7)
            break

    ax.set_xticks([1, 4, 10, 20, 30])
    ax.set_title(f"{ds}  [{cat}]", fontsize=8.5, color=color, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Annotate d_eff numbers at k=1 and k=30
    if rows and rows[0]["k"] == 1:
        ax.annotate(f"d={rows[0]['d_eff']}", xy=(1, rets[0]),
                    xytext=(2, rets[0] + 5), fontsize=6, color="#4488cc")
    if rows and rows[-1]["k"] == 30:
        ax.annotate(f"d={rows[-1]['d_eff']}", xy=(30, rets[-1]),
                    xytext=(21, rets[-1] - 10), fontsize=6, color="#4488cc")

# Common axis labels on edges
for ax in axes2[6:]:
    ax.set_xlabel("k", fontsize=8)

fig2.text(0.01, 0.5, "d_eff retention (%)  [blue, left axis]",
          va="center", rotation="vertical", fontsize=9, color="#4488cc")
fig2.text(0.99, 0.5, "Part A [pp]  [colored, right axis]",
          va="center", rotation="vertical", fontsize=9, ha="right")

fig2.suptitle(
    "Q2: d_eff Retention vs Part A — All 9 Datasets  (fixed splits)\n"
    "Blue=dimension retention, Color=Part A gap, dashed=zero line",
    fontsize=11
)
plt.tight_layout(rect=[0.02, 0, 0.98, 0.94])

for ext in ["pdf", "png"]:
    p = os.path.join(OUTPUT_DIR, f"plot_q2_all_datasets.{ext}")
    fig2.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p}")
plt.close()

# ─── STEP 7: Summary text ─────────────────────────────────────────────────────
print()
print("=" * 70)
print("Q2 SUMMARY")
print("=" * 70)

summary_lines = []

summary_lines.append("Q2: DIMENSION LOSS vs CROSSOVER ANALYSIS")
summary_lines.append("=" * 60)
summary_lines.append(f"Split type: {SPLIT}")
summary_lines.append(f"k values: {AVAILABLE_K}")
summary_lines.append(f"Note: k=15 and k=25 were never run (globally absent)")
summary_lines.append("")

# Group by crossover type
for ctype in ["Crossover", "Near-crossover", "Monotonic"]:
    rows_c = [r for r in corr_rows if r["crossover_type"] == ctype]
    if not rows_c:
        continue
    summary_lines.append(f"--- {ctype} datasets ({len(rows_c)}) ---")
    for r in rows_c:
        pr = r['pearson_r']
        pv = r['p_value']
        r_str = f"{pr:+.3f}" if isinstance(pr, float) else str(pr)
        p_str = f"{pv:.4f}" if isinstance(pv, float) else str(pv)
        summary_lines.append(
            f"  {r['dataset']:25s}: r={r_str}  p={p_str}  "
            f"pattern={r['pattern']}  d_eff_range={r['d_eff_range']}  "
            f"retention@k30={r['d_eff_retention_k30']}"
        )
    summary_lines.append("")

summary_lines.append("--- Crossover detection details ---")
for r in crossover_rows:
    if r["crossover_type"] == "Crossover":
        summary_lines.append(
            f"  {r['dataset']:25s}: Part A crosses 0 at k≈{r['k_crossover']}  "
            f"d_eff≈{r['d_eff_at_crossover']}  retention≈{r['retention_at_crossover']}"
        )
    elif r["crossover_type"] == "Near-crossover":
        summary_lines.append(
            f"  {r['dataset']:25s}: Part A={r['part_a_at_k30']}pp at k=30  "
            f"(no sign change observed up to k=30)"
        )
summary_lines.append("")

# Key finding
summary_lines.append("--- Key finding ---")
cross_rets  = [r["d_eff_retention_k30"] for r in corr_rows
               if r["crossover_type"] == "Crossover"
               and isinstance(r["d_eff_retention_k30"], float)]
mono_rets   = [r["d_eff_retention_k30"] for r in corr_rows
               if r["crossover_type"] == "Monotonic"
               and isinstance(r["d_eff_retention_k30"], float)]
if cross_rets and mono_rets:
    summary_lines.append(
        f"  Mean d_eff retention at k=30:  "
        f"Crossover={np.mean(cross_rets):.3f}  Monotonic={np.mean(mono_rets):.3f}"
    )

cross_rs = [r["pearson_r"] for r in corr_rows
            if r["crossover_type"] == "Crossover" and isinstance(r["pearson_r"], float)]
mono_rs  = [r["pearson_r"] for r in corr_rows
            if r["crossover_type"] == "Monotonic" and isinstance(r["pearson_r"], float)]
if cross_rs and mono_rs:
    summary_lines.append(
        f"  Mean Pearson r (retention vs Part A):  "
        f"Crossover={np.mean(cross_rs):+.3f}  Monotonic={np.mean(mono_rs):+.3f}"
    )
    summary_lines.append(
        "  Interpretation: Positive r for monotonic → both d_eff and Part A move together."
        " Crossover r reflects the transition."
    )

for line in summary_lines:
    print("  " + line if line.startswith("-") else line)

summary_path = os.path.join(OUTPUT_DIR, "q2_summary.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))
print(f"\n  Saved: {summary_path}")

print()
print("Done. All outputs in:", OUTPUT_DIR)
