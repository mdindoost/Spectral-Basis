"""
SpectralGeometryMisalignment/analyze_label_label.py

Post-run analysis for label_label_experiment.py.
Produces: results/label_label/analysis/analysis_report.txt

EXPERIMENT: Full label information (upper bound / ceiling).
  X_label_label: ALL nodes → true one-hot class label (deterministic)
  U_label_label: Rayleigh-Ritz of (L,D) on X_label_label (computed once per dataset)

  *** Scientific probe — NOT a predictive model ***
  Test labels appear in features → this measures the ceiling of what
  GCN(X) / GCN(U) can extract when given perfect label information.

3 models: GCN_X | GCN_U (StandardScaler) | GCN_rowNorm_U (row-norm at layer input)

Reports:
  META  Dataset metadata
  COMP  Completeness check
  R1    Main results table  — canonical config (adam lr=0.01, all 5 split seeds)
  R2    Above-random check  — flag anything below random baseline
  R3    Ceiling gap         — Gap_U = GCN_U − GCN_X (positive = spectral helps at ceiling)
                              Gap_RN = GCN_rowNorm_U − GCN_U
  R4    Optimizer sensitivity — SGD best-LR vs Adam best-LR
  R5    Split seed stability  — variance across 5 random splits
  R6    Training curves       — epoch checkpoints (selected datasets)
  R9    Verified findings

Canonical config: adam, lr=0.01, random splits, all 5 split seeds pooled
  → mean over split_seeds={0..4} × train_seeds={0..4} = up to 25 runs per cell

Best-LR selection: highest mean val_acc_curve[-1] across all seeds at that config.

Usage:
  /home/md724/Spectral-Basis/venv/bin/python \\
      SpectralGeometryMisalignment/analyze_label_label.py
"""

import os, sys, json, math
from datetime import date
import numpy as np
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(_HERE, 'results', 'label_label')
ANALYSIS_DIR = os.path.join(RESULTS_ROOT, 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────

ALL_DATASETS = [
    'cora', 'citeseer', 'pubmed', 'ogbn_arxiv', 'wikics',
    'amazon_computers', 'amazon_photo', 'coauthor_cs', 'coauthor_physics',
]
MODEL_KEYS  = ['GCN_X', 'GCN_U', 'GCN_rowNorm_U']
OPTIMIZERS  = ['sgd', 'adam']
LRS         = [0.001, 0.01, 0.1]
SPLIT_SEEDS = [0, 1, 2, 3, 4]

CANON_OPT        = 'adam'
CANON_LR         = 0.01
# No fixed split for label_label — random only (Yiannis specified).
# Canonical pools all 5 split seeds → 25 runs per cell.

CURVE_DATASETS = ['cora', 'pubmed', 'ogbn_arxiv']
EPOCH_CHECKS   = [50, 100, 200, 300, 400, 500]

# Expected total records per dataset:
# 5 split_seeds × 2 opts × 3 lrs × 5 train_seeds = 150
EXPECTED_RECORDS = 5 * 2 * 3 * 5


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all(datasets=ALL_DATASETS):
    """Load all complete JSON records (all 3 model keys present)."""
    records = defaultdict(list)
    for ds in datasets:
        d = os.path.join(RESULTS_ROOT, ds)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith('.json'):
                continue
            with open(os.path.join(d, fn)) as fh:
                r = json.load(fh)
            if all(k in r for k in MODEL_KEYS):
                records[ds].append(r)
    return records


# ── Filtering helpers ──────────────────────────────────────────────────────────

def filt(recs, split_seed=None, optimizer=None, lr=None):
    out = recs
    if split_seed is not None: out = [r for r in out if r.get('split_seed') == split_seed]
    if optimizer  is not None: out = [r for r in out if r.get('optimizer')  == optimizer]
    if lr         is not None: out = [r for r in out if abs(r.get('lr', 0) - lr) < 1e-9]
    return out


# ── Extraction helpers ─────────────────────────────────────────────────────────

def test_accs(recs, mk):
    return [r[mk]['final_test_acc'] * 100 for r in recs
            if mk in r and not math.isnan(r[mk]['final_test_acc'])]

def val_accs(recs, mk):
    vals = []
    for r in recs:
        if mk in r:
            c = r[mk].get('val_acc_curve', [])
            if c and not math.isnan(c[-1]):
                vals.append(c[-1] * 100)
    return vals

def test_curves(recs, mk):
    curves = []
    for r in recs:
        if mk in r:
            c = r[mk].get('test_acc_curve', [])
            if c and len(c) == 500 and not any(math.isnan(v) for v in c):
                curves.append(np.array(c) * 100)
    return curves

def baseline(recs):
    for r in recs:
        if 'random_baseline' in r:
            return float(r['random_baseline'])
    return float('nan')

def mf(lst): return float(np.mean(lst)) if lst else float('nan')
def sf(lst): return float(np.std(lst))  if lst else float('nan')

def ms(lst):
    if not lst: return '   N/A    '
    return f'{np.mean(lst):6.2f}±{np.std(lst):4.2f}'


# ── Best-LR selector ──────────────────────────────────────────────────────────

def best_lr_for(recs_ds, mk, optimizer):
    """Select LR with highest mean val_acc_curve[-1] across all seeds (all split seeds)."""
    best_lr, best_mean = None, -1.0
    for lr in LRS:
        r  = filt(recs_ds, optimizer=optimizer, lr=lr)
        va = val_accs(r, mk)
        if va and mf(va) > best_mean:
            best_mean = mf(va)
            best_lr   = lr
    return best_lr, best_mean


# ══════════════════════════════════════════════════════════════════════════════
# COMPLETENESS CHECK
# ══════════════════════════════════════════════════════════════════════════════

def completeness_check(records, line):
    line('=' * 80)
    line('COMPLETENESS CHECK')
    line(f'Expected: {EXPECTED_RECORDS} records per dataset')
    line('(5 split_seeds × 2 opts × 3 lrs × 5 train_seeds = 150)')
    line('=' * 80)
    any_incomplete = False
    for ds in ALL_DATASETS:
        n  = len(records.get(ds, []))
        ok = '✓' if n == EXPECTED_RECORDS else f'INCOMPLETE ({n}/{EXPECTED_RECORDS})'
        flag = '  <-- check' if n != EXPECTED_RECORDS else ''
        if n != EXPECTED_RECORDS: any_incomplete = True
        line(f'  {ds:<22}: {ok}{flag}')
    line('')
    if any_incomplete:
        line('WARNING: Some datasets incomplete. Reports show N/A for missing data.')
    else:
        line('All datasets complete.')


# ══════════════════════════════════════════════════════════════════════════════
# DATASET METADATA
# ══════════════════════════════════════════════════════════════════════════════

def dataset_metadata(records, line):
    line('=' * 80)
    line('DATASET METADATA')
    line('=' * 80)
    line(f'{"Dataset":<22} {"n_nodes":>8} {"num_cls":>8} {"input_dim":>10} '
         f'{"d_eff":>6} {"base%":>6} {"n_train":>8} {"records":>8}')
    line('-' * 85)
    for ds in ALL_DATASETS:
        recs = records.get(ds, [])
        if not recs:
            line(f'{ds:<22}  [no data]')
            continue
        r    = recs[0]
        n    = r.get('n_train', 0) + r.get('n_val', 0) + r.get('n_test', 0)
        nc   = r.get('num_classes', 'N/A')
        idim = r.get('input_dim', 'N/A')
        deff = r.get('d_eff', 'N/A')
        base = r.get('random_baseline', float('nan'))
        ntr  = r.get('n_train', 'N/A')
        line(f'{ds:<22} {n:>8,} {str(nc):>8} {str(idim):>10} {str(deff):>6} '
             f'{base:>6.2f} {str(ntr):>8} {len(recs):>8}')
    line('')
    line('input_dim = num_classes (X_label_label has num_classes columns)')
    line('d_eff     = number of Rayleigh-Ritz eigenvectors (≤ num_classes)')
    line('n_train   = training nodes on a random 60/20/20 split (varies by split_seed)')
    line('')
    line('NOTE: X_label_label gives ALL nodes their true one-hot label.')
    line('      This is a scientific probe — test labels appear in features.')
    line('      GCN_X and GCN_U here are NOT valid predictive models.')
    line('      High accuracy reflects label propagation, not generalisation.')


# ══════════════════════════════════════════════════════════════════════════════
# R1 — Main results table
# ══════════════════════════════════════════════════════════════════════════════

def report1(records, line):
    line('=' * 80)
    line('REPORT 1: MAIN RESULTS TABLE')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, all 5 random split seeds pooled')
    line('Mean ± std over split_seeds={0..4} × train_seeds={0..4} (up to 25 runs)')
    line('=' * 80)

    hdr = (f'{"Dataset":<22} {"base%":>5} | {"GCN_X":>13} | {"GCN_U":>13} | '
           f'{"GCN_rowNorm_U":>13}')
    line(hdr)

    for opt in OPTIMIZERS:
        line(f'\n  Optimizer: {opt.upper()}')
        line('  ' + '-' * (len(hdr) - 2))
        for ds in ALL_DATASETS:
            recs = records.get(ds, [])
            if not recs:
                line(f'  {ds:<22}  [no data]')
                continue
            base = baseline(recs)
            cols = []
            for mk in MODEL_KEYS:
                blr, _ = best_lr_for(recs, mk, opt)
                if blr is None:
                    cols.append('   N/A     ')
                    continue
                r  = filt(recs, optimizer=opt, lr=blr)
                ta = test_accs(r, mk)
                cols.append(ms(ta) + f' [lr={blr}]')
            line(f'  {ds:<22} {base:>5.1f} | {cols[0]:>20} | {cols[1]:>20} | {cols[2]:>20}')

    line('')
    line('Best LR selected by mean val_acc_curve[-1] across all split × train seeds.')


# ══════════════════════════════════════════════════════════════════════════════
# R2 — Above-random check
# ══════════════════════════════════════════════════════════════════════════════

def report2(records, line):
    line('=' * 80)
    line('REPORT 2: ABOVE-RANDOM CHECK')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, all split seeds pooled')
    line('Ratio = mean_test_acc / random_baseline  (>1.0 = above random)')
    line('ANOMALY if ratio < 1.0')
    line('=' * 80)

    hdr = (f'{"Dataset":<22} {"base%":>5} | '
           + '  '.join(f'{m:>15}' for m in MODEL_KEYS))
    line(hdr)
    line('-' * len(hdr))

    anomalies = []
    for ds in ALL_DATASETS:
        recs = filt(records.get(ds, []), optimizer=CANON_OPT, lr=CANON_LR)
        base = baseline(recs)
        if not recs or math.isnan(base):
            line(f'{ds:<22}  [no data]')
            continue
        cols = []
        for mk in MODEL_KEYS:
            ta   = test_accs(recs, mk)
            mean = mf(ta)
            if math.isnan(mean):
                cols.append('      N/A      ')
            else:
                ratio = mean / base
                flag  = ' ***' if ratio < 1.0 else ''
                cols.append(f'{mean:5.2f}% ({ratio:.2f}x){flag}')
                if ratio < 1.0:
                    anomalies.append((ds, mk, mean, base, ratio))
        line(f'{ds:<22} {base:>5.1f} | ' + '  '.join(f'{c:>15}' for c in cols))

    line('')
    if anomalies:
        line('ANOMALIES (below random baseline):')
        for ds, mk, mean, base, ratio in anomalies:
            line(f'  [{ds}] {mk}: {mean:.2f}% vs baseline {base:.2f}%  (ratio={ratio:.3f}x)')
    else:
        line('No anomalies — all models at or above random baseline.')
    line('')
    line('REMINDER: High accuracy here (near 100%) is EXPECTED — X_label_label embeds')
    line('  all true labels into features. This experiment measures the ceiling of what')
    line('  GCN(X) vs GCN(U) can achieve given perfect label information.')


# ══════════════════════════════════════════════════════════════════════════════
# R3 — Ceiling gap: Gap_U = GCN_U − GCN_X
# ══════════════════════════════════════════════════════════════════════════════

def report3(records, line):
    line('=' * 80)
    line('REPORT 3: CEILING GAP — Gap_U = GCN_U − GCN_X')
    line('Interpretation: span(X_label_label) = span(U_label_label) — gap is geometric.')
    line('  Gap_U > 0: Rayleigh-Ritz pre-spreads labels globally → GCN(U) easier to optimise.')
    line('  Gap_U < 0: Direct one-hot features easier for GCN than spectral projection.')
    line('  This is the CEILING — both models see perfect label information.')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, all 5 split seeds pooled.')
    line('Mean over split_seeds × train_seeds.')
    line('=' * 80)

    line(f'\n{"Dataset":<22} {"base%":>5} | {"GCN_X":>7} {"GCN_U":>7} '
         f'{"Gap_U":>7} | {"GCN_rnU":>8} {"Gap_RN":>8}')
    line('-' * 72)

    for ds in ALL_DATASETS:
        recs = filt(records.get(ds, []), optimizer=CANON_OPT, lr=CANON_LR)
        base = baseline(recs)
        if not recs:
            line(f'{ds:<22}  [no data]')
            continue

        gcnx  = mf(test_accs(recs, 'GCN_X'))
        gcnu  = mf(test_accs(recs, 'GCN_U'))
        gcnrn = mf(test_accs(recs, 'GCN_rowNorm_U'))
        gap_u  = gcnu  - gcnx
        gap_rn = gcnrn - gcnu

        def f(v): return f'{v:7.2f}' if not math.isnan(v) else '    N/A'
        marker_u  = ' *** (U helps)'  if gap_u  > 5  else (' *** (X better)' if gap_u  < -5  else '')
        marker_rn = ' *** (rn helps)' if gap_rn > 5  else (' *** (rn hurts)' if gap_rn < -5  else '')

        line(f'{ds:<22} {base:>5.1f} | {f(gcnx)} {f(gcnu)} {gap_u:>+7.2f}{marker_u} | '
             f'{f(gcnrn)} {gap_rn:>+8.2f}{marker_rn}')

    line('')
    line('Gap_U  = GCN_U − GCN_X        (cost/benefit of spectral projection at ceiling)')
    line('Gap_RN = GCN_rowNorm_U − GCN_U (additional effect of row-norm on spectral GCN)')
    line('')
    line('Compare Gap_U here to the same gap in label_rand (Exp 1) and gcn_rand (Exp 0)')
    line('to trace how the spectral-projection advantage evolves with label information.')


# ══════════════════════════════════════════════════════════════════════════════
# R4 — Optimizer sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def report4(records, line):
    line('=' * 80)
    line('REPORT 4: OPTIMIZER SENSITIVITY')
    line('Best LR per optimizer | Mean±std across all split_seeds × train_seeds')
    line('=' * 80)

    for mk in MODEL_KEYS:
        line(f'\n  Model: {mk}')
        line(f'  {"Dataset":<22} {"base%":>5} | {"SGD best-LR":>18} | '
             f'{"Adam best-LR":>18} | {"Gap (Adam−SGD)":>15}')
        line('  ' + '-' * 80)
        for ds in ALL_DATASETS:
            recs = records.get(ds, [])
            base = baseline(recs)
            cols = {}
            for opt in OPTIMIZERS:
                blr, _ = best_lr_for(recs, mk, opt)
                if blr is None:
                    cols[opt] = ('   N/A     ', float('nan'))
                    continue
                r  = filt(recs, optimizer=opt, lr=blr)
                ta = test_accs(r, mk)
                cols[opt] = (ms(ta) + f'[lr={blr}]', mf(ta))
            gap   = cols['adam'][1] - cols['sgd'][1]
            gap_s = f'{gap:>+8.2f} pp' if not math.isnan(gap) else '   N/A    '
            line(f'  {ds:<22} {base:>5.1f} | {cols["sgd"][0]:>18} | '
                 f'{cols["adam"][0]:>18} | {gap_s:>15}')


# ══════════════════════════════════════════════════════════════════════════════
# R5 — Split seed stability
# ══════════════════════════════════════════════════════════════════════════════

def report5(records, line):
    line('=' * 80)
    line('REPORT 5: SPLIT SEED STABILITY')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}')
    line('Per-seed mean computed over train_seeds={0..4} (5 runs).')
    line('split_std = std of per-seed means.  seed_std = mean of per-seed stds.')
    line('Stable: split_std << seed_std  (split choice does not dominate variance).')
    line('NOTE: U_label_label is fixed — only StandardScaler changes across splits.')
    line('      X_label_label is also fixed (all labels). Variance comes from train nodes.')
    line('=' * 80)

    for mk in MODEL_KEYS:
        line(f'\n  Model: {mk}')
        line(f'  {"Dataset":<22} '
             + ' '.join(f'{"ss"+str(s):>8}' for s in SPLIT_SEEDS)
             + f' | {"split_std":>10} {"seed_std":>10} | {"stable?":>8}')
        line('  ' + '-' * (22 + 9 * len(SPLIT_SEEDS) + 35))

        for ds in ALL_DATASETS:
            recs = records.get(ds, [])
            seed_means, seed_stds, seed_strs = [], [], []
            for ss in SPLIT_SEEDS:
                r  = filt(recs, split_seed=ss, optimizer=CANON_OPT, lr=CANON_LR)
                ta = test_accs(r, mk)
                if ta:
                    seed_means.append(mf(ta))
                    seed_stds.append(sf(ta))
                    seed_strs.append(f'{mf(ta):6.2f}')
                else:
                    seed_strs.append('   N/A')

            if len(seed_means) < 2:
                line(f'  {ds:<22}  [insufficient data]')
                continue

            split_std = float(np.std(seed_means))
            seed_std  = float(np.mean(seed_stds))
            stable    = 'YES' if (seed_std > 0 and split_std < seed_std) else 'NO'
            vals_str  = ' '.join(f'{s:>8}' for s in seed_strs)
            line(f'  {ds:<22} {vals_str} | {split_std:>10.3f} {seed_std:>10.3f} | {stable:>8}')


# ══════════════════════════════════════════════════════════════════════════════
# R6 — Training curves
# ══════════════════════════════════════════════════════════════════════════════

def report6(records, line):
    line('=' * 80)
    line('REPORT 6: TRAINING CURVES — EPOCH CHECKPOINTS')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, all 5 split seeds pooled')
    line('Values = mean test accuracy (%) across split_seeds × train_seeds.')
    line(f'Datasets shown: {CURVE_DATASETS}')
    line('=' * 80)

    for ds in CURVE_DATASETS:
        recs = filt(records.get(ds, []), optimizer=CANON_OPT, lr=CANON_LR)
        base = baseline(recs)
        line(f'\n  Dataset: {ds}  (random baseline = {base:.2f}%)')
        line(f'  {"Model":<16}' + ''.join(f'  ep{ep:>3}' for ep in EPOCH_CHECKS))
        line('  ' + '-' * (16 + 8 * len(EPOCH_CHECKS)))
        for mk in MODEL_KEYS:
            curves = test_curves(recs, mk)
            if not curves:
                line(f'  {mk:<16}  [no data]')
                continue
            arr  = np.stack(curves).mean(axis=0)
            vals = ''.join(f'  {arr[ep - 1]:>5.2f}' for ep in EPOCH_CHECKS)
            line(f'  {mk:<16}{vals}')


# ══════════════════════════════════════════════════════════════════════════════
# R9 — Verified findings
# ══════════════════════════════════════════════════════════════════════════════

def report_findings(records, line):
    line('=' * 80)
    line('REPORT 9: VERIFIED FINDINGS')
    line('Numbers first, conclusions second.')
    line(f'Canonical = {CANON_OPT}, lr={CANON_LR}, all 5 split seeds pooled.')
    line('=' * 80)

    def f2(v): return f'{v:.2f}' if not math.isnan(v) else 'N/A'

    # Pre-compute canonical means (all splits pooled)
    canon = {}
    for ds in ALL_DATASETS:
        recs = filt(records.get(ds, []), optimizer=CANON_OPT, lr=CANON_LR)
        canon[ds] = {mk: mf(test_accs(recs, mk)) for mk in MODEL_KEYS}
        canon[ds]['base'] = baseline(recs)

    # Finding 1: GCN_X ceiling — how high does direct one-hot labeling get GCN?
    line('')
    line('─' * 78)
    line('FINDING 1: GCN_X ceiling accuracy (all nodes given true one-hot labels)')
    line('─' * 78)
    line('GCN_X here is effectively transductive label propagation via message passing.')
    line('Expected: near-perfect for datasets with high label homophily.')
    line('')
    gcnx_vals = {}
    for ds in ALL_DATASETS:
        gcnx = canon[ds]['GCN_X']
        base = canon[ds]['base']
        ratio = gcnx / base if not math.isnan(base) and base > 0 else float('nan')
        gcnx_vals[ds] = gcnx
        marker = f'  ({ratio:.1f}× above random)' if not math.isnan(ratio) else ''
        line(f'  {ds:<22}: {f2(gcnx)}%{marker}')

    valid = [v for v in gcnx_vals.values() if not math.isnan(v)]
    if valid:
        line(f'  Range: {min(valid):.2f}% to {max(valid):.2f}%')

    # Finding 2: Ceiling gap
    line('')
    line('─' * 78)
    line('FINDING 2: Ceiling gap — Gap_U = GCN_U − GCN_X at full label information')
    line('─' * 78)
    line('span(X_label_label) = span(U_label_label) → gap is geometric, not informational.')
    line('Gap_U > 0: Rayleigh-Ritz globally pre-distributes label structure → GCN(U) easier.')
    line('Gap_U < 0: Direct one-hot features more advantageous for local GCN aggregation.')
    line('')

    gap_u_vals = {}
    for ds in ALL_DATASETS:
        gcnx = canon[ds]['GCN_X']
        gcnu = canon[ds]['GCN_U']
        gap  = gcnu - gcnx
        gap_u_vals[ds] = gap
        marker = ' *** positive (U helps)'  if gap > 2 else \
                 (' *** negative (X better)' if gap < -2 else ' (near-zero)')
        line(f'  {ds:<22}: GCN_X={f2(gcnx)}%  GCN_U={f2(gcnu)}%  Gap_U={gap:>+.2f} pp{marker}')

    valid_gaps = [v for v in gap_u_vals.values() if not math.isnan(v)]
    pos  = [ds for ds, v in gap_u_vals.items() if v > 2]
    neg  = [ds for ds, v in gap_u_vals.items() if v < -2]
    line('')
    if valid_gaps:
        line(f'  Range: {min(valid_gaps):+.2f} to {max(valid_gaps):+.2f} pp')
        if pos: line(f'  U helps  (Gap_U > +2pp): {pos}')
        if neg: line(f'  X better (Gap_U < −2pp): {neg}')

    # Finding 3: Row-norm at ceiling
    line('')
    line('─' * 78)
    line('FINDING 3: Row-norm effect at ceiling (Gap_RN = GCN_rowNorm_U − GCN_U)')
    line('─' * 78)
    line('')
    gap_rn_vals = {}
    for ds in ALL_DATASETS:
        gcnu  = canon[ds]['GCN_U']
        gcnrn = canon[ds]['GCN_rowNorm_U']
        gap   = gcnrn - gcnu
        gap_rn_vals[ds] = gap
        marker = ' *** (helps)' if gap > 2 else (' *** (hurts)' if gap < -2 else '')
        line(f'  {ds:<22}: GCN_U={f2(gcnu)}%  GCN_rowNorm_U={f2(gcnrn)}%  '
             f'Gap_RN={gap:>+.2f} pp{marker}')

    valid_rn = [v for v in gap_rn_vals.values() if not math.isnan(v)]
    if valid_rn:
        line(f'  Range: {min(valid_rn):+.2f} to {max(valid_rn):+.2f} pp')

    # Finding 4: Split seed stability summary
    line('')
    line('─' * 78)
    line('FINDING 4: Split seed stability — does which 60% we train on matter?')
    line('─' * 78)
    line('NOTE: X_label_label and U_label_label are deterministic (do not change with split).')
    line('  Variance across splits = which nodes are in train/val/test, not feature variance.')
    line('')
    line(f'  {"Dataset":<22} {"GCN_X split_std":>16} {"GCN_U split_std":>16} '
         f'{"GCN_rnU split_std":>18}')
    line('  ' + '-' * 76)
    for ds in ALL_DATASETS:
        stds = []
        for mk in MODEL_KEYS:
            per_seed = []
            for ss in SPLIT_SEEDS:
                r  = filt(records.get(ds, []), split_seed=ss,
                          optimizer=CANON_OPT, lr=CANON_LR)
                ta = test_accs(r, mk)
                if ta:
                    per_seed.append(mf(ta))
            stds.append(f'{float(np.std(per_seed)):.3f}' if len(per_seed) >= 2 else '  N/A ')
        line(f'  {ds:<22} {stds[0]:>16} {stds[1]:>16} {stds[2]:>18}')

    # Summary
    line('')
    line('─' * 78)
    line('SUMMARY — KEY NUMBERS')
    line('─' * 78)
    vg = [v for v in gap_u_vals.values()  if not math.isnan(v)]
    vr = [v for v in gap_rn_vals.values() if not math.isnan(v)]
    vx = [v for v in gcnx_vals.values()   if not math.isnan(v)]
    if vx: line(f'  GCN_X ceiling range:    {min(vx):.2f}% to {max(vx):.2f}%')
    if vg: line(f'  Gap_U  range (canonical): {min(vg):+.2f} to {max(vg):+.2f} pp')
    if vr: line(f'  Gap_RN range (canonical): {min(vr):+.2f} to {max(vr):+.2f} pp')
    line('  All numbers [VERIFIED] from canonical config (adam lr=0.01, all split seeds).')
    line('')
    line('SCIENTIFIC NOTE:')
    line('  These results represent the upper bound on what GCN can extract from the')
    line('  label structure. Because test labels are embedded in features, accuracy')
    line('  reflects label propagation via message passing — not generalisation.')
    line('  The key scientific question is not "how high is accuracy?" but rather')
    line('  "does span(X)=span(U) at ceiling produce the same geometric gap as')
    line('  at partial (label_rand) and zero (gcn_rand) label information?"')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    report_path = os.path.join(ANALYSIS_DIR, 'analysis_report.txt')
    lines_buf   = []

    def line(s=''):
        print(s)
        lines_buf.append(s)

    line('=' * 80)
    line('EXPERIMENT 2: FULL LABEL INFORMATION (CEILING) — FULL ANALYSIS REPORT')
    line('SpectralGeometryMisalignment / label_label_experiment.py')
    line(f'Date: {date.today().strftime("%B %Y")}')
    line('=' * 80)
    line('')
    line('EXPERIMENTAL SETUP')
    line('------------------')
    line('X_label_label: ALL nodes → their true one-hot class label (deterministic).')
    line('U_label_label: Rayleigh-Ritz of (L,D) on X_label_label. Computed once per dataset.')
    line('span(X) = span(U) — gap between GCN(X) and GCN(U) is optimization geometry.')
    line('')
    line('*** Scientific probe — test labels appear in features. NOT a predictive model. ***')
    line('GCN(X_label_label) = transductive label propagation via message passing.')
    line('Upper bound: measures ceiling performance + geometry of spectral projection.')
    line('')
    line('3 models:  GCN_X | GCN_U (StandardScaler) | GCN_rowNorm_U (row-norm at layer input)')
    line('Training:  SGD + Adam, LR∈{0.001,0.01,0.1}, 500 epochs, WD=0')
    line('Seeds:     5 train × 5 random split seeds (60/20/20) = 25 runs per (opt,lr,model)')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, all 5 split seeds pooled')
    line('')

    print('Loading results...')
    records = load_all()
    line('')

    completeness_check(records, line);  line('')
    dataset_metadata(records, line);    line('')
    report2(records, line);             line('')  # above-random (sanity gate)
    report1(records, line);             line('')  # main table
    report3(records, line);             line('')  # ceiling gap
    report4(records, line);             line('')  # optimizer sensitivity
    report5(records, line);             line('')  # split seed stability
    report6(records, line);             line('')  # training curves
    report_findings(records, line)

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
