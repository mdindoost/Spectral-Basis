"""
SpectralGeometryMisalignment/analyze_label_combined.py

Combined information-ladder analysis across all experiments.
Produces: results/combined/analysis_report.txt

THE THREE-LEVEL INFORMATION LADDER
───────────────────────────────────
  Level 0 — gcn_rand    : Zero class info   (X_rand ~ N(0,1))
  Level 1 — label_rand  : Partial class info (train → one-hot, val/test → random)
  Level 2 — label_label : Full class info   (all nodes → one-hot, ceiling / upper bound)

At each level we compare:
  GCN_X         → GCN on raw features X
  GCN_U         → GCN on Rayleigh-Ritz eigenvectors U (+ StandardScaler)
  GCN_rowNorm_U → GCN on U with row-norm at each layer input (Option B)

  span(X) = span(U) at every level → any gap is optimization geometry, not information.

KEY COMPARISONS
  Gap_U    = GCN_U − GCN_X            (benefit of spectral projection over direct features)
  Gap_RN   = GCN_rowNorm_U − GCN_U    (additional effect of row-norm on top of U)
  Gap_U_T  = Transformer_U − Trans_X  (same gap for Transformer architecture — Level 2 only)

ADDITIONAL SECTIONS
  ROB   Level 1 robustness: Gaussian (v1) vs Simplex (v2) random features
        Scientific question: is Gap_U stable across different uninformative feature distributions?
  TRANS Transformer at Level 2 (label_label_v2): Gap_U_Trans vs Gap_U_GCN
        Scientific question: does spectral projection help Transformer the same way it helps GCN?

Reports:
  L0    Level 0 summary loaded from gcn_rand/summary.json
  L1    Level 1 summary (label_rand v1 — Gaussian, canonical)
  L2    Level 2 summary (label_label v1 — GCN, canonical)
  LAD   Ladder table — Gap_U evolution across all 3 levels per dataset
  RN    Row-norm effect evolution across levels
  ROB   Level 1 robustness: v1 (Gaussian) vs v2 (Simplex)
  TRANS Transformer Gap_U at Level 2 vs GCN Gap_U
  REG   Dataset regime analysis
  VF    Verified findings

Canonical configs:
  Level 0: adam, lr=0.01, fixed split, split_seed=0   (gcn_rand convention)
  Level 1: adam, lr=0.01, random split, split_seed=0  (label_rand convention)
  Level 2: adam, lr=0.01, all 5 random split seeds    (label_label convention)
  Transformer (L2): adam, lr=0.01, all 5 random split seeds (label_label_v2)

Data sources:
  Level 0 GCN:         results/gcn_rand/summary.json
  Level 1 v1 (Gaussian): results/label_rand/
  Level 1 v2 (Simplex):  results/label_rand_v2/
  Level 2 GCN:           results/label_label/      ← clean complete v1 run
  Level 2 Transformer:   results/label_label_v2/   ← v2 run with Transformer

Usage:
  /home/md724/Spectral-Basis/venv/bin/python \\
      SpectralGeometryMisalignment/analyze_label_combined.py
"""

import os, sys, json, math
from datetime import date
import numpy as np
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────

_HERE          = os.path.dirname(os.path.abspath(__file__))
GCNRAND_DIR    = os.path.join(_HERE, 'results', 'gcn_rand')
LABRAND_DIR    = os.path.join(_HERE, 'results', 'label_rand')
LABRAND_V2_DIR = os.path.join(_HERE, 'results', 'label_rand_v2')
LABLAB_DIR     = os.path.join(_HERE, 'results', 'label_label')
LABLAB_V2_DIR  = os.path.join(_HERE, 'results', 'label_label_v2')
COMBINED_DIR   = os.path.join(_HERE, 'results', 'combined')
os.makedirs(COMBINED_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────

ALL_DATASETS = [
    'cora', 'citeseer', 'pubmed', 'ogbn_arxiv', 'wikics',
    'amazon_computers', 'amazon_photo', 'coauthor_cs', 'coauthor_physics',
]

# Level 0 model keys (gcn_rand experiment)
L0_KEYS = ['GCN_X', 'GCN_rand', 'GCN_rowNorm_Y', 'MLP_Y', 'MLP_rowNorm_Y']

# Level 1 & 2 GCN model keys
L12_KEYS = ['GCN_X', 'GCN_U', 'GCN_rowNorm_U']

# Level 2 Transformer model keys
TRANS_KEYS = ['Transformer_X', 'Transformer_U', 'Transformer_rowNorm_U']

OPTIMIZERS  = ['sgd', 'adam']
LRS         = [0.001, 0.01, 0.1]
SPLIT_SEEDS = [0, 1, 2, 3, 4]
XRAND_SEEDS = [0, 1, 2]

# Canonical configs per level
L0_CANON = dict(optimizer='adam', lr=0.01, split_type='fixed', split_seed=0)
L1_CANON = dict(optimizer='adam', lr=0.01, split_type='random', split_seed=0)
# L2: adam, lr=0.01, all 5 split seeds pooled (no split_seed filter)
L2_CANON = dict(optimizer='adam', lr=0.01)


# ── Helpers ────────────────────────────────────────────────────────────────────

def mf(lst): return float(np.mean(lst)) if lst else float('nan')
def sf(lst): return float(np.std(lst))  if lst else float('nan')
def ms(lst):
    if not lst: return '   N/A    '
    return f'{np.mean(lst):6.2f}±{np.std(lst):4.2f}'
def f2(v):   return f'{v:.2f}' if not math.isnan(v) else ' N/A'
def fp(v):   return f'{v:+.2f}' if not math.isnan(v) else '  N/A'


# ── Level 0 loading (from summary.json) ───────────────────────────────────────

def load_l0_summary():
    """Load gcn_rand summary.json (Option B: use pre-aggregated results)."""
    path = os.path.join(GCNRAND_DIR, 'summary.json')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def l0_canonical_acc(summary, ds, mk):
    """
    Extract canonical (adam, lr=0.01, fixed split) mean from gcn_rand summary.json.

    Actual structure:
      summary[ds]['fixed']['adam']['0.01'] = { 'GCN_X_mean': ..., 'GCN_rand_mean': ..., ... }
    Values are already in percentage (0–100 scale).

    Returns (mean, std) in percent, or (nan, nan) if not found.
    """
    if summary is None:
        return float('nan'), float('nan')
    try:
        cfg  = summary[ds]['fixed']['adam']['0.01']
        mean = float(cfg[f'{mk}_mean'])
        std  = float(cfg.get(f'{mk}_std', 0.0))
        return mean, std
    except (KeyError, TypeError):
        return float('nan'), float('nan')


# ── Level 1 loading ───────────────────────────────────────────────────────────

def load_l1():
    """Load label_rand records."""
    records = defaultdict(list)
    for ds in ALL_DATASETS:
        d = os.path.join(LABRAND_DIR, ds)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith('.json'):
                continue
            with open(os.path.join(d, fn)) as fh:
                r = json.load(fh)
            if all(k in r for k in L12_KEYS):
                records[ds].append(r)
    return records


def filt_l1(recs, split_type=None, split_seed=None, xrand_seed=None,
             optimizer=None, lr=None):
    out = recs
    if split_type is not None: out = [r for r in out if r.get('split_type') == split_type]
    if split_seed is not None: out = [r for r in out if r.get('split_seed') == split_seed]
    if xrand_seed is not None: out = [r for r in out if r.get('xrand_seed') == xrand_seed]
    if optimizer  is not None: out = [r for r in out if r.get('optimizer')  == optimizer]
    if lr         is not None: out = [r for r in out if abs(r.get('lr', 0) - lr) < 1e-9]
    return out


def l1_canonical_acc(records, ds, mk):
    """adam, lr=0.01, random split, split_seed=0 — mean over xrand × train seeds."""
    recs = filt_l1(records.get(ds, []),
                   split_type='random', split_seed=0, optimizer='adam', lr=0.01)
    vals = [r[mk]['final_test_acc'] * 100 for r in recs
            if mk in r and not math.isnan(r[mk]['final_test_acc'])]
    return mf(vals), sf(vals)


# ── Level 2 loading ───────────────────────────────────────────────────────────

def load_l2():
    """Load label_label records."""
    records = defaultdict(list)
    for ds in ALL_DATASETS:
        d = os.path.join(LABLAB_DIR, ds)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith('.json'):
                continue
            with open(os.path.join(d, fn)) as fh:
                r = json.load(fh)
            if all(k in r for k in L12_KEYS):
                records[ds].append(r)
    return records


def filt_l2(recs, split_seed=None, optimizer=None, lr=None):
    out = recs
    if split_seed is not None: out = [r for r in out if r.get('split_seed') == split_seed]
    if optimizer  is not None: out = [r for r in out if r.get('optimizer')  == optimizer]
    if lr         is not None: out = [r for r in out if abs(r.get('lr', 0) - lr) < 1e-9]
    return out


def l2_canonical_acc(records, ds, mk):
    """adam, lr=0.01, all 5 split seeds pooled — mean over split × train seeds."""
    recs = filt_l2(records.get(ds, []), optimizer='adam', lr=0.01)
    vals = [r[mk]['final_test_acc'] * 100 for r in recs
            if mk in r and not math.isnan(r[mk]['final_test_acc'])]
    return mf(vals), sf(vals)


# ── Level 1 v2 loading (label_rand_v2 — simplex features) ────────────────────

def load_l1v2():
    """Load label_rand_v2 records (val/test on probability simplex)."""
    records = defaultdict(list)
    for ds in ALL_DATASETS:
        d = os.path.join(LABRAND_V2_DIR, ds)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith('.json'):
                continue
            with open(os.path.join(d, fn)) as fh:
                r = json.load(fh)
            if all(k in r and isinstance(r.get(k), dict) for k in L12_KEYS):
                records[ds].append(r)
    return records


def l1v2_canonical_acc(records, ds, mk):
    """adam, lr=0.01, random split, split_seed=0 — mean over xrand × train seeds (v2)."""
    recs = filt_l1(records.get(ds, []),
                   split_type='random', split_seed=0, optimizer='adam', lr=0.01)
    vals = [r[mk]['final_test_acc'] * 100 for r in recs
            if mk in r and isinstance(r[mk], dict) and not math.isnan(r[mk]['final_test_acc'])]
    return mf(vals), sf(vals)


# ── Level 2 v2 loading (label_label_v2 — GCN + Transformer) ──────────────────

def load_l2v2():
    """Load label_label_v2 records. Each record may have GCN and/or Transformer keys."""
    records = defaultdict(list)
    for ds in ALL_DATASETS:
        d = os.path.join(LABLAB_V2_DIR, ds)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith('.json'):
                continue
            with open(os.path.join(d, fn)) as fh:
                r = json.load(fh)
            records[ds].append(r)
    return records


def _is_valid_result(entry):
    """True if entry is a dict with a finite final_test_acc (not skipped, not None)."""
    return (isinstance(entry, dict)
            and not entry.get('skipped', False)
            and 'final_test_acc' in entry
            and not math.isnan(entry['final_test_acc']))


def l2v2_gcn_acc(records, ds, mk):
    """GCN canonical acc from label_label_v2: adam, lr=0.01, all 5 split seeds."""
    recs = filt_l2(records.get(ds, []), optimizer='adam', lr=0.01)
    vals = [r[mk]['final_test_acc'] * 100 for r in recs
            if mk in r and _is_valid_result(r.get(mk))]
    return mf(vals), sf(vals)


def l2v2_transformer_acc(records, ds, mk):
    """
    Transformer canonical acc from label_label_v2: adam, lr=0.01, all 5 split seeds.
    Returns (mean, std, status) where status is 'ok', 'skipped', or 'N/A'.
    """
    recs = filt_l2(records.get(ds, []), optimizer='adam', lr=0.01)
    if not recs:
        return float('nan'), float('nan'), 'N/A'
    # Check feasibility flag (any record)
    for r in recs:
        if r.get('transformer_feasible') is False:
            return float('nan'), float('nan'), 'skipped'
    vals = [r[mk]['final_test_acc'] * 100 for r in recs
            if mk in r and _is_valid_result(r.get(mk))]
    if not vals:
        return float('nan'), float('nan'), 'N/A'
    return mf(vals), sf(vals), 'ok'


# ── Random baseline ────────────────────────────────────────────────────────────

def get_baseline(records, ds):
    for r in records.get(ds, []):
        if 'random_baseline' in r:
            return float(r['random_baseline'])
    return float('nan')


# ══════════════════════════════════════════════════════════════════════════════
# L0 — Level 0 summary
# ══════════════════════════════════════════════════════════════════════════════

def level0_summary(l0_sum, l2_recs, line):
    """Level 0 summary: gcn_rand canonical results."""
    line('=' * 80)
    line('LEVEL 0: ZERO CLASS INFORMATION — gcn_rand canonical results')
    line('  X_rand ~ N(0,1), zero class info, labels and topology unchanged.')
    line('  Canonical: adam, lr=0.01, fixed split, split_seed=0')
    line('  Source: gcn_rand/summary.json')
    line('=' * 80)

    if l0_sum is None:
        line('  ERROR: gcn_rand/summary.json not found.')
        line('  Run analyze_gcn_rand.py first, or check that the file exists.')
        return

    # Show: GCN_X, GCN_rand, GCN_rowNorm_Y, MLP_Y
    show_keys = ['GCN_X', 'GCN_rand', 'GCN_rowNorm_Y', 'MLP_Y']
    line(f'\n  {"Dataset":<22} {"base%":>5} | '
         + ' | '.join(f'{k:>14}' for k in show_keys))
    line('  ' + '-' * (22 + 6 + 3 + 17 * len(show_keys)))

    for ds in ALL_DATASETS:
        base = get_baseline(l2_recs, ds)  # use l2 for baseline (all share same graph)
        # Try to get baseline from summary
        if math.isnan(base):
            try:
                base = float(l0_sum[ds].get('random_baseline',
                             l0_sum[ds].get('base', float('nan'))))
                if base <= 1.5: base *= 100
            except (KeyError, TypeError):
                pass
        cols = []
        for mk in show_keys:
            mean, std = l0_canonical_acc(l0_sum, ds, mk)
            if math.isnan(mean):
                cols.append('     N/A      ')
            else:
                cols.append(f'{mean:6.2f}±{std:4.2f}')
        base_s = f'{base:.1f}' if not math.isnan(base) else ' N/A'
        line(f'  {ds:<22} {base_s:>5} | ' + ' | '.join(f'{c:>14}' for c in cols))

    line('')
    line('  GCN_X        = GCN on real features X (feature ceiling at Level 0)')
    line('  GCN_rand     = GCN on X_rand — class info from topology only')
    line('  GCN_rowNorm_Y= GCN on Y_rand with row-norm at each layer input (Option B)')
    line('  MLP_Y        = linear classifier on Y_rand + StandardScaler')


# ══════════════════════════════════════════════════════════════════════════════
# L1 — Level 1 summary
# ══════════════════════════════════════════════════════════════════════════════

def level1_summary(l1_recs, line):
    line('=' * 80)
    line('LEVEL 1: PARTIAL CLASS INFORMATION — label_rand canonical results')
    line('  X_label_rand: train → one-hot, val/test → N(0,1).')
    line('  Canonical: adam, lr=0.01, random split, split_seed=0')
    line('  Mean over xrand_seeds={0,1,2} × train_seeds={0..4} (up to 15 runs)')
    line('=' * 80)

    line(f'\n  {"Dataset":<22} {"base%":>5} | {"GCN_X":>13} | {"GCN_U":>13} | '
         f'{"GCN_rowNorm_U":>13}')
    line('  ' + '-' * 72)
    for ds in ALL_DATASETS:
        base = get_baseline(l1_recs, ds)
        cols = []
        for mk in L12_KEYS:
            mean, std = l1_canonical_acc(l1_recs, ds, mk)
            cols.append(f'{mean:6.2f}±{std:4.2f}' if not math.isnan(mean) else '     N/A  ')
        base_s = f'{base:.1f}' if not math.isnan(base) else ' N/A'
        line(f'  {ds:<22} {base_s:>5} | {cols[0]:>13} | {cols[1]:>13} | {cols[2]:>13}')


# ══════════════════════════════════════════════════════════════════════════════
# L2 — Level 2 summary
# ══════════════════════════════════════════════════════════════════════════════

def level2_summary(l2_recs, line):
    line('=' * 80)
    line('LEVEL 2: FULL CLASS INFORMATION (CEILING) — label_label canonical results')
    line('  X_label_label: ALL nodes → true one-hot label (deterministic).')
    line('  *** Scientific probe — test labels in features. NOT predictive. ***')
    line('  Canonical: adam, lr=0.01, all 5 split seeds pooled')
    line('  Mean over split_seeds={0..4} × train_seeds={0..4} (up to 25 runs)')
    line('=' * 80)

    line(f'\n  {"Dataset":<22} {"base%":>5} | {"GCN_X":>13} | {"GCN_U":>13} | '
         f'{"GCN_rowNorm_U":>13}')
    line('  ' + '-' * 72)
    for ds in ALL_DATASETS:
        base = get_baseline(l2_recs, ds)
        cols = []
        for mk in L12_KEYS:
            mean, std = l2_canonical_acc(l2_recs, ds, mk)
            cols.append(f'{mean:6.2f}±{std:4.2f}' if not math.isnan(mean) else '     N/A  ')
        base_s = f'{base:.1f}' if not math.isnan(base) else ' N/A'
        line(f'  {ds:<22} {base_s:>5} | {cols[0]:>13} | {cols[1]:>13} | {cols[2]:>13}')


# ══════════════════════════════════════════════════════════════════════════════
# LAD — Ladder table: Gap_U evolution across levels
# ══════════════════════════════════════════════════════════════════════════════

def ladder_table(l0_sum, l1_recs, l2_recs, line):
    line('=' * 80)
    line('LADDER TABLE: Gap_U = GCN_U − GCN_X across information levels')
    line('─────────────────────────────────────────────────────────────')
    line('  Level 0 (gcn_rand):    GCN_rand − GCN_X  (topology vs real features)')
    line('  Level 1 (label_rand):  GCN_U   − GCN_X  (partial label info)')
    line('  Level 2 (label_label): GCN_U   − GCN_X  (full label info / ceiling)')
    line('  Positive = spectral / topology advantage. Negative = direct features win.')
    line('NOTE: Level 0 compares GCN_rand (not GCN_U) to GCN_X — different model role.')
    line('      Levels 1 and 2 are the clean span(X)=span(U) geometry comparison.')
    line('=' * 80)

    line(f'\n  {"Dataset":<22} {"base%":>5} | {"Gap L0":>9} {"Gap L1":>9} {"Gap L2":>9} | '
         f'{"trend":>12}')
    line('  ' + '-' * 76)

    gap_data = {}
    for ds in ALL_DATASETS:
        base = get_baseline(l2_recs, ds)
        if math.isnan(base):
            base = get_baseline(l1_recs, ds)

        # Level 0: GCN_rand − GCN_X
        gcnx_l0, _  = l0_canonical_acc(l0_sum, ds, 'GCN_X')   if l0_sum else (float('nan'), float('nan'))
        gcnr_l0, _  = l0_canonical_acc(l0_sum, ds, 'GCN_rand') if l0_sum else (float('nan'), float('nan'))
        gap_l0 = gcnr_l0 - gcnx_l0

        # Level 1: GCN_U − GCN_X
        gcnx_l1, _ = l1_canonical_acc(l1_recs, ds, 'GCN_X')
        gcnu_l1, _ = l1_canonical_acc(l1_recs, ds, 'GCN_U')
        gap_l1 = gcnu_l1 - gcnx_l1

        # Level 2: GCN_U − GCN_X
        gcnx_l2, _ = l2_canonical_acc(l2_recs, ds, 'GCN_X')
        gcnu_l2, _ = l2_canonical_acc(l2_recs, ds, 'GCN_U')
        gap_l2 = gcnu_l2 - gcnx_l2

        gap_data[ds] = (gap_l0, gap_l1, gap_l2)

        # Trend: L1→L2 only (span(X)=span(U) comparison; L0 is a different model)
        if not math.isnan(gap_l1) and not math.isnan(gap_l2):
            if gap_l2 > gap_l1:
                trend = 'L1→L2 incr'
            elif gap_l2 < gap_l1:
                trend = 'L1→L2 decr'
            else:
                trend = 'L1→L2 flat'
        else:
            trend = 'N/A'

        base_s = f'{base:.1f}' if not math.isnan(base) else ' N/A'
        line(f'  {ds:<22} {base_s:>5} | {fp(gap_l0):>9} {fp(gap_l1):>9} {fp(gap_l2):>9} | '
             f'{trend:>12}')

    line('')
    line('  Interpretation guide (L1→L2 trend only; L0 uses a different model — GCN_rand):')
    line('  L1→L2 incr → spectral projection advantage grows with more label info.')
    line('  L1→L2 decr → spectral advantage erodes as direct features improve.')
    line('  L1→L2 flat → no change between partial and full label information.')
    return gap_data


# ══════════════════════════════════════════════════════════════════════════════
# RN — Row-norm evolution across levels
# ══════════════════════════════════════════════════════════════════════════════

def rownorm_evolution(l0_sum, l1_recs, l2_recs, line):
    line('=' * 80)
    line('ROW-NORM EVOLUTION: Gap_RN = GCN_rowNorm_U − GCN_U across levels')
    line('  Level 0: GCN_rowNorm_Y − GCN_rand  (row-norm on Y_rand vs plain X_rand GCN)')
    line('  Level 1: GCN_rowNorm_U − GCN_U     (partial label info)')
    line('  Level 2: GCN_rowNorm_U − GCN_U     (full label info / ceiling)')
    line('  Positive = row-norm helps. Negative = row-norm hurts.')
    line('=' * 80)

    line(f'\n  {"Dataset":<22} {"base%":>5} | {"RN L0":>9} {"RN L1":>9} {"RN L2":>9}')
    line('  ' + '-' * 60)

    for ds in ALL_DATASETS:
        base = get_baseline(l2_recs, ds)
        if math.isnan(base):
            base = get_baseline(l1_recs, ds)

        # Level 0: GCN_rowNorm_Y − GCN_rand
        gcnr_l0, _ = l0_canonical_acc(l0_sum, ds, 'GCN_rand')      if l0_sum else (float('nan'), float('nan'))
        gcnrn_l0,_ = l0_canonical_acc(l0_sum, ds, 'GCN_rowNorm_Y') if l0_sum else (float('nan'), float('nan'))
        rn_l0 = gcnrn_l0 - gcnr_l0

        # Level 1: GCN_rowNorm_U − GCN_U
        gcnu_l1, _  = l1_canonical_acc(l1_recs, ds, 'GCN_U')
        gcnrn_l1, _ = l1_canonical_acc(l1_recs, ds, 'GCN_rowNorm_U')
        rn_l1 = gcnrn_l1 - gcnu_l1

        # Level 2: GCN_rowNorm_U − GCN_U
        gcnu_l2, _  = l2_canonical_acc(l2_recs, ds, 'GCN_U')
        gcnrn_l2, _ = l2_canonical_acc(l2_recs, ds, 'GCN_rowNorm_U')
        rn_l2 = gcnrn_l2 - gcnu_l2

        base_s = f'{base:.1f}' if not math.isnan(base) else ' N/A'
        line(f'  {ds:<22} {base_s:>5} | {fp(rn_l0):>9} {fp(rn_l1):>9} {fp(rn_l2):>9}')

    line('')
    line('  A consistent sign across all 3 levels = row-norm has a stable effect on')
    line('  this dataset regardless of how much label info the features carry.')
    line('  Sign flip = row-norm interaction depends on the label-feature geometry.')


# ══════════════════════════════════════════════════════════════════════════════
# REG — Dataset regime analysis
# ══════════════════════════════════════════════════════════════════════════════

def regime_analysis(gap_data, l2_recs, l1_recs, line):
    line('=' * 80)
    line('DATASET REGIME ANALYSIS')
    line('  Classifying each dataset by Gap_U sign at Levels 1 and 2.')
    line('  Gap_U > +2pp = "U helps";  Gap_U < −2pp = "X better";  else "near-parity".')
    line('=' * 80)

    regimes = {}
    for ds in ALL_DATASETS:
        _, gap_l1, gap_l2 = gap_data.get(ds, (float('nan'), float('nan'), float('nan')))

        def classify(v):
            if math.isnan(v): return 'N/A'
            if v > 2:  return 'U_helps'
            if v < -2: return 'X_better'
            return 'near-parity'

        r1, r2 = classify(gap_l1), classify(gap_l2)
        regimes[ds] = (r1, r2)

    line(f'\n  {"Dataset":<22} {"L1 regime":>14} {"L2 regime":>14} {"Consistency":>14}')
    line('  ' + '-' * 68)
    for ds in ALL_DATASETS:
        r1, r2 = regimes.get(ds, ('N/A', 'N/A'))
        consistent = 'consistent' if r1 == r2 else 'regime_shift'
        line(f'  {ds:<22} {r1:>14} {r2:>14} {consistent:>14}')

    line('')
    # Group by L1 regime
    for regime in ['U_helps', 'X_better', 'near-parity']:
        datasets_in_regime = [ds for ds, (r1, r2) in regimes.items() if r1 == regime]
        if datasets_in_regime:
            line(f'  L1 {regime}: {datasets_in_regime}')

    line('')
    line('  Regime shift (L1 vs L2) highlights datasets where the advantage of spectral')
    line('  projection changes qualitatively when features become fully label-informative.')


# ══════════════════════════════════════════════════════════════════════════════
# VF — Verified findings
# ══════════════════════════════════════════════════════════════════════════════

def verified_findings(l0_sum, l1_recs, l2_recs, gap_data, line):
    line('=' * 80)
    line('VERIFIED FINDINGS — COMBINED LADDER ANALYSIS')
    line('Numbers first, conclusions second.')
    line('All numbers [VERIFIED] from canonical configs (see header for details).')
    line('=' * 80)

    def f2(v): return f'{v:.2f}' if not math.isnan(v) else 'N/A'

    # Collect all gap values
    all_gap_l1 = {ds: gap_data[ds][1] for ds in ALL_DATASETS if ds in gap_data}
    all_gap_l2 = {ds: gap_data[ds][2] for ds in ALL_DATASETS if ds in gap_data}

    valid_l1 = [(ds, v) for ds, v in all_gap_l1.items() if not math.isnan(v)]
    valid_l2 = [(ds, v) for ds, v in all_gap_l2.items() if not math.isnan(v)]

    # Finding 1: Gap_U distribution at L1 and L2
    line('')
    line('─' * 78)
    line('FINDING 1: Gap_U distribution at Level 1 (partial) and Level 2 (ceiling)')
    line('─' * 78)
    line('')
    if valid_l1:
        vals_l1 = [v for _, v in valid_l1]
        pos_l1  = [(ds, v) for ds, v in valid_l1 if v > 2]
        neg_l1  = [(ds, v) for ds, v in valid_l1 if v < -2]
        line(f'  Level 1  range: {min(vals_l1):+.2f} to {max(vals_l1):+.2f} pp')
        if pos_l1: line(f'    U helps  (>+2pp): {[ds for ds,_ in pos_l1]} '
                        f'  max={max(v for _,v in pos_l1):+.2f}pp')
        if neg_l1: line(f'    X better (<−2pp): {[ds for ds,_ in neg_l1]} '
                        f'  min={min(v for _,v in neg_l1):+.2f}pp')
    if valid_l2:
        vals_l2 = [v for _, v in valid_l2]
        pos_l2  = [(ds, v) for ds, v in valid_l2 if v > 2]
        neg_l2  = [(ds, v) for ds, v in valid_l2 if v < -2]
        line(f'  Level 2  range: {min(vals_l2):+.2f} to {max(vals_l2):+.2f} pp')
        if pos_l2: line(f'    U helps  (>+2pp): {[ds for ds,_ in pos_l2]} '
                        f'  max={max(v for _,v in pos_l2):+.2f}pp')
        if neg_l2: line(f'    X better (<−2pp): {[ds for ds,_ in neg_l2]} '
                        f'  min={min(v for _,v in neg_l2):+.2f}pp')

    # Finding 2: L1→L2 trend (span(X)=span(U) levels only; L0 uses GCN_rand ≠ GCN_U)
    line('')
    line('─' * 78)
    line('FINDING 2: Gap_U trend from Level 1 → Level 2')
    line('  (L0 excluded: GCN_rand vs GCN_X is NOT a span(X)=span(U) comparison)')
    line('─' * 78)
    line('')
    l1l2_incr, l1l2_decr, l1l2_flat = [], [], []
    for ds in ALL_DATASETS:
        if ds not in gap_data: continue
        _, g1, g2 = gap_data[ds]
        if math.isnan(g1) or math.isnan(g2): continue
        if g2 > g1:
            l1l2_incr.append(ds)
        elif g2 < g1:
            l1l2_decr.append(ds)
        else:
            l1l2_flat.append(ds)
    if l1l2_decr:
        line(f'  L1→L2 Decreasing — Gap_U more negative at ceiling ({len(l1l2_decr)}/{len(ALL_DATASETS)}):')
        line(f'    {l1l2_decr}')
    if l1l2_incr:
        line(f'  L1→L2 Increasing — Gap_U improves at ceiling ({len(l1l2_incr)}/{len(ALL_DATASETS)}):')
        line(f'    {l1l2_incr}')
    if l1l2_flat:
        line(f'  L1→L2 Flat — no change ({len(l1l2_flat)}/{len(ALL_DATASETS)}): {l1l2_flat}')
    line('')
    line('  Decreasing trend = spectral advantage (if any at L1) erodes under full info.')
    line('  Interpretation: at ceiling GCN_X is harder to beat because direct features')
    line('  perfectly encode class structure — U adds no new information but the geometry')
    line('  of U still imposes optimization overhead.')

    # Finding 3: Row-norm consistency
    line('')
    line('─' * 78)
    line('FINDING 3: Row-norm effect consistency across levels')
    line('─' * 78)
    line('')
    rn_consistent, rn_flip = [], []
    for ds in ALL_DATASETS:
        # Level 1 rn gap
        gcnu_l1, _  = l1_canonical_acc(l1_recs, ds, 'GCN_U')
        gcnrn_l1, _ = l1_canonical_acc(l1_recs, ds, 'GCN_rowNorm_U')
        rn1 = gcnrn_l1 - gcnu_l1
        # Level 2 rn gap
        gcnu_l2, _  = l2_canonical_acc(l2_recs, ds, 'GCN_U')
        gcnrn_l2, _ = l2_canonical_acc(l2_recs, ds, 'GCN_rowNorm_U')
        rn2 = gcnrn_l2 - gcnu_l2
        if not math.isnan(rn1) and not math.isnan(rn2):
            if (rn1 > 0) == (rn2 > 0):
                rn_consistent.append(f'{ds}(L1={rn1:+.1f},L2={rn2:+.1f})')
            else:
                rn_flip.append(f'{ds}(L1={rn1:+.1f},L2={rn2:+.1f})')
    if rn_consistent:
        line(f'  Consistent sign (L1 & L2):  {rn_consistent}')
    if rn_flip:
        line(f'  Sign flip (L1→L2):          {rn_flip}')
    line('')
    line('  Datasets where row-norm flips sign = the effect depends on how much label')
    line('  information the spectral features carry, not just the spectral geometry.')

    # Summary
    line('')
    line('─' * 78)
    line('SUMMARY — CROSS-EXPERIMENT KEY NUMBERS')
    line('─' * 78)
    line('  All values at canonical configs (see script header).')
    line('  [VERIFIED] from raw per-run JSON files (L1, L2) and summary.json (L0).')
    line('')
    line(f'  {"Dataset":<22} | {"L0 GCN_rand":>10} {"L1 GCN_X":>10} {"L2 GCN_X":>10} |'
         f' {"Gap L1":>7} {"Gap L2":>7}')
    line('  ' + '-' * 78)
    for ds in ALL_DATASETS:
        gcnr_l0, _ = l0_canonical_acc(l0_sum, ds, 'GCN_rand') if l0_sum else (float('nan'), float('nan'))
        gcnx_l1, _ = l1_canonical_acc(l1_recs, ds, 'GCN_X')
        gcnx_l2, _ = l2_canonical_acc(l2_recs, ds, 'GCN_X')
        _, gap_l1, gap_l2 = gap_data.get(ds, (float('nan'), float('nan'), float('nan')))
        line(f'  {ds:<22} | {f2(gcnr_l0):>10} {f2(gcnx_l1):>10} {f2(gcnx_l2):>10} |'
             f' {fp(gap_l1):>7} {fp(gap_l2):>7}')


# ══════════════════════════════════════════════════════════════════════════════
# ROB — Level 1 robustness: v1 (Gaussian) vs v2 (Simplex)
# ══════════════════════════════════════════════════════════════════════════════

def level1_robustness(l1_recs, l1v2_recs, line):
    line('=' * 80)
    line('LEVEL 1 ROBUSTNESS: Gaussian (v1) vs Simplex (v2) Random Features')
    line('─────────────────────────────────────────────────────────────────')
    line('  v1: val/test ~ N(0,1) Gaussian — features not on simplex')
    line('  v2: val/test ~ Uniform probability simplex — rows sum to 1, entries ≥ 0')
    line('  Scientific question: is Gap_U stable across different uninformative')
    line('  feature distributions? If yes → conclusion robust to feature geometry.')
    line('  Canonical: adam, lr=0.01, random split, split_seed=0.')
    line('  Gap_U = GCN_U − GCN_X  (span(X)=span(U) → geometry, not information).')
    line('=' * 80)

    line(f'\n  {"Dataset":<22} {"base%":>5} | {"GCN_X v1":>9} {"GCN_U v1":>9} {"Gap v1":>7} |'
         f' {"GCN_X v2":>9} {"GCN_U v2":>9} {"Gap v2":>7} | {"Δ(v2−v1)":>9} {"robust?":>8}')
    line('  ' + '-' * 103)

    for ds in ALL_DATASETS:
        base = get_baseline(l1_recs, ds)
        if math.isnan(base):
            base = get_baseline(l1v2_recs, ds)

        x1, _ = l1_canonical_acc(l1_recs, ds, 'GCN_X')
        u1, _ = l1_canonical_acc(l1_recs, ds, 'GCN_U')
        x2, _ = l1v2_canonical_acc(l1v2_recs, ds, 'GCN_X')
        u2, _ = l1v2_canonical_acc(l1v2_recs, ds, 'GCN_U')

        gap1 = u1 - x1
        gap2 = u2 - x2
        delta = gap2 - gap1

        # Robust: same sign or both near-parity (|gap| < 2pp)
        def classify(v):
            if math.isnan(v): return 'N/A'
            if v > 2:  return '+'
            if v < -2: return '-'
            return '~0'
        r1, r2 = classify(gap1), classify(gap2)
        robust = 'YES' if r1 == r2 else 'NO (sign flip)'

        base_s = f'{base:.1f}' if not math.isnan(base) else ' N/A'
        line(f'  {ds:<22} {base_s:>5} | {f2(x1):>9} {f2(u1):>9} {fp(gap1):>7} |'
             f' {f2(x2):>9} {f2(u2):>9} {fp(gap2):>7} | {fp(delta):>9} {robust:>8}')

    line('')
    line('  robust? YES = Gap_U sign agrees between v1 and v2 (or both near-parity).')
    line('  Δ(v2−v1): how much Gap_U shifts when switching from Gaussian to Simplex.')
    line('  A large |Δ| with sign flip = Gap_U is sensitive to feature geometry.')
    line('  A small |Δ| consistent sign = robust conclusion across feature distributions.')


# ══════════════════════════════════════════════════════════════════════════════
# TRANS — Transformer Gap_U at Level 2 (label_label_v2)
# ══════════════════════════════════════════════════════════════════════════════

def transformer_level2(l2v2_recs, l2_recs, line):
    line('=' * 80)
    line('TRANSFORMER AT LEVEL 2 (label_label_v2)')
    line('────────────────────────────────────────')
    line('  Architecture comparison at the full-information ceiling.')
    line('  GCN_U uses Rayleigh-Ritz U + message passing via Â.')
    line('  Transformer_U uses Rayleigh-Ritz U + global self-attention (no graph Â).')
    line('  span(X) = span(U) → any Gap_U is optimization geometry, not information.')
    line('  skip = dataset too large for O(N²) attention (N > 25000).')
    line('  Canonical: adam, lr=0.01, all 5 random split seeds pooled.')
    line('=' * 80)

    line(f'\n  {"Dataset":<22} {"base%":>5} |'
         f' {"GCN_X":>8} {"GCN_U":>8} {"Gap_GCN":>8} |'
         f' {"Trans_X":>9} {"Trans_U":>9} {"Gap_Trans":>10} |'
         f' {"Δ(T−G)":>7}')
    line('  ' + '-' * 97)

    gap_gcn_all   = {}
    gap_trans_all = {}
    trans_raw_acc = {}   # ds -> (tx, tu) — for ceiling detection

    for ds in ALL_DATASETS:
        base = get_baseline(l2_recs, ds)
        if math.isnan(base):
            base = get_baseline(l2v2_recs, ds)

        # GCN from v2 (consistent source for this comparison)
        gcn_x, _  = l2v2_gcn_acc(l2v2_recs, ds, 'GCN_X')
        gcn_u, _  = l2v2_gcn_acc(l2v2_recs, ds, 'GCN_U')
        gap_gcn   = gcn_u - gcn_x
        gap_gcn_all[ds] = gap_gcn

        # Transformer
        tx, _, st_x = l2v2_transformer_acc(l2v2_recs, ds, 'Transformer_X')
        tu, _, st_u = l2v2_transformer_acc(l2v2_recs, ds, 'Transformer_U')

        if st_x == 'skipped':
            trans_x_s = '    skip'
            trans_u_s = '    skip'
            gap_trans_s = '      skip'
            delta_s = '   skip'
            gap_trans_all[ds] = float('nan')
        elif st_x == 'N/A' or math.isnan(tx) or math.isnan(tu):
            trans_x_s = '     N/A'
            trans_u_s = '     N/A'
            gap_trans_s = '       N/A'
            delta_s = '    N/A'
            gap_trans_all[ds] = float('nan')
        else:
            gap_trans = tu - tx
            gap_trans_all[ds] = gap_trans
            trans_raw_acc[ds] = (tx, tu)
            trans_x_s = f'{tx:8.2f}'
            trans_u_s = f'{tu:8.2f}'
            gap_trans_s = f'{gap_trans:+9.2f}'
            delta = gap_trans - gap_gcn
            delta_s = f'{delta:+6.2f}'

        base_s = f'{base:.1f}' if not math.isnan(base) else ' N/A'
        line(f'  {ds:<22} {base_s:>5} |'
             f' {f2(gcn_x):>8} {f2(gcn_u):>8} {fp(gap_gcn):>8} |'
             f' {trans_x_s:>9} {trans_u_s:>9} {gap_trans_s:>10} |'
             f' {delta_s:>7}')

    line('')
    line('  Gap_GCN   = GCN_U   − GCN_X      (spectral projection effect on GCN)')
    line('  Gap_Trans = Trans_U  − Trans_X    (spectral projection effect on Transformer)')
    line('  Δ(T−G)    = Gap_Trans − Gap_GCN   (does U help Transformer more/less than GCN?)')
    line('')

    # Summary: compare signs across datasets
    both_valid = {ds for ds in ALL_DATASETS
                  if not math.isnan(gap_gcn_all.get(ds, float('nan')))
                  and not math.isnan(gap_trans_all.get(ds, float('nan')))}
    if both_valid:
        # Ceiling case: both Trans_X and Trans_U hit ~100% → gap_trans ≈ 0 by saturation
        # This is physically different from a genuine negative gap (features win over U).
        def _is_ceiling(ds):
            if ds not in trans_raw_acc: return False
            tx, tu = trans_raw_acc[ds]
            return tx >= 99.9 and tu >= 99.9

        ceiling_ds = [ds for ds in both_valid if _is_ceiling(ds)]
        diff_sign  = [ds for ds in both_valid
                      if not _is_ceiling(ds)
                      and (gap_gcn_all[ds] > 0) != (gap_trans_all[ds] > 0)]
        same_sign  = [ds for ds in both_valid
                      if not _is_ceiling(ds) and ds not in diff_sign]

        gaps_gcn_v   = [gap_gcn_all[ds]   for ds in both_valid]
        gaps_trans_v = [gap_trans_all[ds]  for ds in both_valid if not _is_ceiling(ds)]
        line(f'  Datasets with both GCN and Transformer results: {sorted(both_valid)}')
        line(f'  Gap_GCN   range (these datasets):  {min(gaps_gcn_v):+.2f} to {max(gaps_gcn_v):+.2f} pp')
        if gaps_trans_v:
            line(f'  Gap_Trans range (non-ceiling):     {min(gaps_trans_v):+.2f} to {max(gaps_trans_v):+.2f} pp')
        if ceiling_ds:
            line(f'  Ceiling (Trans_X=Trans_U≈100%, gap=0 by saturation): {ceiling_ds}')
            line(f'    NOTE: gap_trans=0.00 here is a ceiling artifact, not genuine parity with GCN.')
        if diff_sign:
            line(f'  Different sign — architecture × spectral interaction: {diff_sign}')
        if same_sign:
            line(f'  Same Gap sign (GCN & Trans agree): {same_sign}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    report_path = os.path.join(COMBINED_DIR, 'analysis_report.txt')
    lines_buf   = []

    def line(s=''):
        print(s)
        lines_buf.append(s)

    line('=' * 80)
    line('COMBINED INFORMATION-LADDER ANALYSIS')
    line('SpectralGeometryMisalignment — gcn_rand / label_rand (v1+v2) / label_label (v1+v2)')
    line(f'Date: {date.today().strftime("%B %Y")}')
    line('=' * 80)
    line('')
    line('INFORMATION LADDER')
    line('------------------')
    line('  Level 0 — gcn_rand    : X_rand ~ N(0,1)            Zero class info')
    line('  Level 1 — label_rand  : train→1-hot, val/test→rand  Partial class info')
    line('  Level 2 — label_label : all nodes→1-hot (ceiling)   Full class info')
    line('')
    line('  Levels 1 & 2: span(X) = span(U) → Gap_U is optimization geometry, not info loss.')
    line('  Level 0: GCN_rand vs GCN_X — NOT span-equivalent; topology signal comparison.')
    line('  3 models: GCN_X | GCN_U (StandardScaler) | GCN_rowNorm_U (row-norm Option B)')
    line('')
    line('CANONICAL CONFIGS')
    line('  Level 0: adam, lr=0.01, fixed split, split_seed=0  (gcn_rand convention)')
    line('  Level 1: adam, lr=0.01, random split, split_seed=0 (label_rand convention)')
    line('  Level 2: adam, lr=0.01, all 5 random split seeds   (label_label convention)')
    line('')

    print('Loading results...')
    l0_sum    = load_l0_summary()
    l1_recs   = load_l1()
    l1v2_recs = load_l1v2()
    l2_recs   = load_l2()
    l2v2_recs = load_l2v2()

    if l0_sum is None:
        line('WARNING: gcn_rand/summary.json not found.')
        line('  Level 0 rows will show N/A.')
        line('  To generate: run analyze_gcn_rand.py (which writes summary.json).')
    line('')

    # Count available data
    l1_ds   = [ds for ds in ALL_DATASETS if l1_recs.get(ds)]
    l1v2_ds = [ds for ds in ALL_DATASETS if l1v2_recs.get(ds)]
    l2_ds   = [ds for ds in ALL_DATASETS if l2_recs.get(ds)]
    l2v2_ds = [ds for ds in ALL_DATASETS if l2v2_recs.get(ds)]
    line(f'Level 0 (gcn_rand):        {"loaded" if l0_sum else "MISSING"} (summary.json)')
    line(f'Level 1 v1 (label_rand):   {len(l1_ds)}/{len(ALL_DATASETS)} datasets — Gaussian features')
    line(f'Level 1 v2 (label_rand_v2):{len(l1v2_ds)}/{len(ALL_DATASETS)} datasets — Simplex features')
    line(f'Level 2 v1 (label_label):  {len(l2_ds)}/{len(ALL_DATASETS)} datasets — GCN only')
    line(f'Level 2 v2 (label_label_v2):{len(l2v2_ds)}/{len(ALL_DATASETS)} datasets — GCN + Transformer')
    line('')

    level0_summary(l0_sum, l2_recs, line);                       line('')
    level1_summary(l1_recs, line);                                line('')
    level2_summary(l2_recs, line);                                line('')
    gap_data = ladder_table(l0_sum, l1_recs, l2_recs, line);     line('')
    rownorm_evolution(l0_sum, l1_recs, l2_recs, line);           line('')
    level1_robustness(l1_recs, l1v2_recs, line);                 line('')
    transformer_level2(l2v2_recs, l2_recs, line);                line('')
    regime_analysis(gap_data, l2_recs, l1_recs, line);           line('')
    verified_findings(l0_sum, l1_recs, l2_recs, gap_data, line)

    line('')
    line('=' * 80)
    line('END OF REPORT')
    line(f'Output: {report_path}')
    line('=' * 80)

    with open(report_path, 'w') as fh:
        fh.write('\n'.join(lines_buf) + '\n')
    print(f'\nReport written → {report_path}')


if __name__ == '__main__':
    main()
