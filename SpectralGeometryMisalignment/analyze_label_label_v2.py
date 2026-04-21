"""
SpectralGeometryMisalignment/analyze_label_label_v2.py

Post-run analysis for label_label_v2_experiment.py.
Produces: results/label_label_v2/analysis/analysis_report.txt

EXPERIMENT: Full label information — v2 (GCN + Transformer comparison)
  X_label_label: ALL nodes → true one-hot label. Deterministic.
  U_label_label: Rayleigh-Ritz of (L,D) on X_label_label. Computed once.
  6 models: GCN_X, GCN_U, GCN_rowNorm_U, Transformer_X, Transformer_U,
            Transformer_rowNorm_U

  Transformer models are SKIPPED for datasets with n > 25000 nodes
  (O(N^2) full self-attention infeasible). Those entries have skipped=True.

Reports:
  META   Dataset metadata + Transformer feasibility
  COMP   Completeness check
  R1     Main results table — all 6 models (canonical: adam lr=0.01, all splits)
  R2     Above-random check
  R3     Ceiling gap — Gap_U = GCN_U − GCN_X  (geometry, not information)
                       Gap_U_T = Transformer_U − Transformer_X
  R4     GCN vs Transformer — core architectural comparison
         Gap_arch_X = Transformer_X − GCN_X  (which architecture benefits more
         Gap_arch_U = Transformer_U − GCN_U   from X vs U features?)
  R5     Row-norm effect — GCN_rowNorm_U − GCN_U vs Transformer_rowNorm_U − Transformer_U
  R6     Optimizer sensitivity
  R7     Split seed stability
  R8     Training curves
  R9     Verified findings

Canonical config: adam, lr=0.01, all 5 split seeds pooled → up to 25 runs per cell.
Skipped Transformer entries are excluded from all statistics (not counted as 0).

Usage:
  /home/md724/Spectral-Basis/venv/bin/python \\
      SpectralGeometryMisalignment/analyze_label_label_v2.py
"""

import os, sys, json, math
from datetime import date
import numpy as np
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(_HERE, 'results', 'label_label_v2')
ANALYSIS_DIR = os.path.join(RESULTS_ROOT, 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────

ALL_DATASETS = [
    'cora', 'citeseer', 'pubmed', 'ogbn_arxiv', 'wikics',
    'amazon_computers', 'amazon_photo', 'coauthor_cs', 'coauthor_physics',
]
GCN_KEYS   = ['GCN_X', 'GCN_U', 'GCN_rowNorm_U']
TRANS_KEYS = ['Transformer_X', 'Transformer_U', 'Transformer_rowNorm_U']
MODEL_KEYS = GCN_KEYS + TRANS_KEYS

OPTIMIZERS  = ['sgd', 'adam']
LRS         = [0.001, 0.01, 0.1]
SPLIT_SEEDS = [0, 1, 2, 3, 4]

CANON_OPT = 'adam'
CANON_LR  = 0.01

CURVE_DATASETS = ['cora', 'pubmed', 'wikics']
EPOCH_CHECKS   = [50, 100, 200, 300, 400, 500]

# 5 split_seeds × 2 opts × 3 lrs × 5 train_seeds = 150
EXPECTED_RECORDS = 5 * 2 * 3 * 5


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all(datasets=ALL_DATASETS):
    """Load records. A record is complete if all GCN keys are present."""
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
            # Complete if all 3 GCN keys present (Transformer may be skipped)
            if all(k in r for k in GCN_KEYS):
                records[ds].append(r)
    return records


# ── Filtering / extraction helpers ────────────────────────────────────────────

def filt(recs, split_seed=None, optimizer=None, lr=None):
    out = recs
    if split_seed is not None: out = [r for r in out if r.get('split_seed') == split_seed]
    if optimizer  is not None: out = [r for r in out if r.get('optimizer')  == optimizer]
    if lr         is not None: out = [r for r in out if abs(r.get('lr',0)-lr) < 1e-9]
    return out


def is_skipped(r, mk):
    """True if model mk was skipped in record r (large graph)."""
    return isinstance(r.get(mk), dict) and r[mk].get('skipped', False)


def test_accs(recs, mk):
    """Final test accuracy (%) — excludes skipped entries."""
    return [r[mk]['final_test_acc']*100 for r in recs
            if mk in r and not is_skipped(r, mk)
            and not math.isnan(r[mk]['final_test_acc'])]


def val_accs(recs, mk):
    vals = []
    for r in recs:
        if mk in r and not is_skipped(r, mk):
            c = r[mk].get('val_acc_curve', [])
            if c and not math.isnan(c[-1]):
                vals.append(c[-1]*100)
    return vals


def test_curves(recs, mk):
    curves = []
    for r in recs:
        if mk in r and not is_skipped(r, mk):
            c = r[mk].get('test_acc_curve', [])
            if c and len(c)==500 and not any(math.isnan(v) for v in c):
                curves.append(np.array(c)*100)
    return curves


def baseline(recs):
    for r in recs:
        if 'random_baseline' in r:
            return float(r['random_baseline'])
    return float('nan')


def transformer_feasible(recs):
    """Check if Transformer was run (not skipped) for this dataset."""
    for r in recs:
        if 'Transformer_X' in r:
            return not is_skipped(r, 'Transformer_X')
    return False


def mf(lst): return float(np.mean(lst)) if lst else float('nan')
def sf(lst): return float(np.std(lst))  if lst else float('nan')
def ms(lst):
    if not lst: return '   N/A    '
    return f'{np.mean(lst):6.2f}±{np.std(lst):4.2f}'
def f2(v):   return f'{v:.2f}' if not math.isnan(v) else ' N/A'
def fp(v):   return f'{v:+.2f}' if not math.isnan(v) else '  N/A'


def best_lr_for(recs_ds, mk, optimizer):
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
    line(f'Expected: {EXPECTED_RECORDS} records per dataset (GCN models)')
    line('(5 split_seeds × 2 opts × 3 lrs × 5 train_seeds = 150)')
    line('Transformer models are skipped for n > 25000 — this is expected.')
    line('=' * 80)
    for ds in ALL_DATASETS:
        recs = records.get(ds, [])
        n    = len(recs)
        ok   = '✓' if n == EXPECTED_RECORDS else f'INCOMPLETE ({n}/{EXPECTED_RECORDS})'
        flag = '  <-- check' if n != EXPECTED_RECORDS else ''
        t_ok = '  Transformer: feasible' if transformer_feasible(recs) \
               else '  Transformer: skipped (n too large)'
        line(f'  {ds:<22}: {ok}{flag}{t_ok}')


# ══════════════════════════════════════════════════════════════════════════════
# DATASET METADATA
# ══════════════════════════════════════════════════════════════════════════════

def dataset_metadata(records, line):
    line('=' * 80)
    line('DATASET METADATA')
    line('=' * 80)
    line(f'{"Dataset":<22} {"n_nodes":>8} {"num_cls":>8} {"d_eff":>6} '
         f'{"base%":>6} {"n_train":>8} {"Transformer":>12} {"records":>8}')
    line('-' * 85)
    for ds in ALL_DATASETS:
        recs = records.get(ds, [])
        if not recs:
            line(f'{ds:<22}  [no data]'); continue
        r    = recs[0]
        n    = r.get('n_train',0) + r.get('n_val',0) + r.get('n_test',0)
        nc   = r.get('num_classes','N/A')
        deff = r.get('d_eff','N/A')
        base = r.get('random_baseline', float('nan'))
        ntr  = r.get('n_train','N/A')
        t_s  = 'YES' if transformer_feasible(recs) else 'SKIPPED'
        line(f'{ds:<22} {n:>8,} {str(nc):>8} {str(deff):>6} '
             f'{base:>6.2f} {str(ntr):>8} {t_s:>12} {len(recs):>8}')
    line('')
    line('Transformer SKIPPED = n > 25000 nodes (O(N^2) attention infeasible).')
    line('NOTE: ALL models use test labels in features — not a predictive experiment.')


# ══════════════════════════════════════════════════════════════════════════════
# R1 — Main results table (all 6 models)
# ══════════════════════════════════════════════════════════════════════════════

def report1(records, line):
    line('=' * 80)
    line('REPORT 1: MAIN RESULTS TABLE — ALL 6 MODELS')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, all 5 split seeds pooled (up to 25 runs)')
    line('skip = Transformer not run (n > 25000)')
    line('=' * 80)

    for opt in OPTIMIZERS:
        line(f'\n  Optimizer: {opt.upper()}')
        line(f'  {"Dataset":<22} {"base%":>5} | '
             f'{"GCN_X":>13} {"GCN_U":>13} {"GCN_rnU":>13} | '
             f'{"Trans_X":>13} {"Trans_U":>13} {"T_rnU":>13}')
        line('  ' + '-' * 100)
        for ds in ALL_DATASETS:
            recs = records.get(ds, [])
            if not recs:
                line(f'  {ds:<22}  [no data]'); continue
            base = baseline(recs)
            cols = []
            for mk in MODEL_KEYS:
                blr, _ = best_lr_for(recs, mk, opt)
                if blr is None:
                    cols.append('   skip  '); continue
                r  = filt(recs, optimizer=opt, lr=blr)
                ta = test_accs(r, mk)
                if not ta:
                    cols.append('   skip  ')
                else:
                    cols.append(ms(ta))
            line(f'  {ds:<22} {base:>5.1f} | '
                 f'{cols[0]:>13} {cols[1]:>13} {cols[2]:>13} | '
                 f'{cols[3]:>13} {cols[4]:>13} {cols[5]:>13}')
    line('')
    line('Best LR by mean val_acc_curve[-1] across all split × train seeds.')


# ══════════════════════════════════════════════════════════════════════════════
# R2 — Above-random check
# ══════════════════════════════════════════════════════════════════════════════

def report2(records, line):
    line('=' * 80)
    line('REPORT 2: ABOVE-RANDOM CHECK')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, all split seeds pooled')
    line('Ratio = mean / random_baseline  (>1.0 = above random)')
    line('REMINDER: near-100% is EXPECTED — test labels are in features.')
    line('=' * 80)
    anomalies = []
    line(f'{"Dataset":<22} {"base%":>5} | '
         + '  '.join(f'{m[:8]:>12}' for m in MODEL_KEYS))
    line('-' * 110)
    for ds in ALL_DATASETS:
        recs = filt(records.get(ds,[]), optimizer=CANON_OPT, lr=CANON_LR)
        base = baseline(recs)
        if not recs or math.isnan(base):
            line(f'{ds:<22}  [no data]'); continue
        cols = []
        for mk in MODEL_KEYS:
            ta   = test_accs(recs, mk)
            if not ta:
                cols.append('    skip    '); continue
            mean  = mf(ta)
            ratio = mean / base
            flag  = ' **' if ratio < 1.0 else ''
            cols.append(f'{mean:5.1f}({ratio:.2f}x){flag}')
            if ratio < 1.0:
                anomalies.append((ds, mk, mean, base, ratio))
        line(f'{ds:<22} {base:>5.1f} | ' + '  '.join(f'{c:>12}' for c in cols))
    line('')
    if anomalies:
        line('ANOMALIES:')
        for ds, mk, mean, base, ratio in anomalies:
            line(f'  [{ds}] {mk}: {mean:.2f}% vs {base:.2f}%  (ratio={ratio:.3f}x)')
    else:
        line('No anomalies.')


# ══════════════════════════════════════════════════════════════════════════════
# R3 — Ceiling gap: Gap_U for GCN and Transformer separately
# ══════════════════════════════════════════════════════════════════════════════

def report3(records, line):
    line('=' * 80)
    line('REPORT 3: CEILING GAP — Gap_U = model(U) − model(X)')
    line('  GCN:         Gap_U_GCN   = GCN_U − GCN_X')
    line('  Transformer: Gap_U_Trans = Transformer_U − Transformer_X')
    line('span(X) = span(U) → gaps are optimization geometry, not information.')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, all 5 split seeds pooled.')
    line('=' * 80)

    line(f'\n{"Dataset":<22} {"base%":>5} | {"GCN_X":>7} {"GCN_U":>7} '
         f'{"Gap_GCN":>8} | {"T_X":>7} {"T_U":>7} {"Gap_T":>8}')
    line('-' * 78)

    for ds in ALL_DATASETS:
        recs  = filt(records.get(ds,[]), optimizer=CANON_OPT, lr=CANON_LR)
        base  = baseline(recs)
        if not recs:
            line(f'{ds:<22}  [no data]'); continue

        gcnx  = mf(test_accs(recs, 'GCN_X'))
        gcnu  = mf(test_accs(recs, 'GCN_U'))
        tx    = mf(test_accs(recs, 'Transformer_X'))
        tu    = mf(test_accs(recs, 'Transformer_U'))
        gap_g = gcnu - gcnx
        gap_t = tu   - tx

        def f(v): return f'{v:7.2f}' if not math.isnan(v) else '   skip'
        def g(v): return f'{v:>+8.2f}' if not math.isnan(v) else '    skip'
        mg = ' ***' if abs(gap_g) > 5 else ''
        mt = ' ***' if abs(gap_t) > 5 else ''
        line(f'{ds:<22} {base:>5.1f} | {f(gcnx)} {f(gcnu)} {g(gap_g)}{mg} | '
             f'{f(tx)} {f(tu)} {g(gap_t)}{mt}')

    line('')
    line('*** = |gap| > 5pp. Positive = U helps, negative = X better.')
    line('Compare Gap_GCN vs Gap_T: does spectral projection help both architectures')
    line('the same way, or differently?')


# ══════════════════════════════════════════════════════════════════════════════
# R4 — GCN vs Transformer: architectural comparison
# ══════════════════════════════════════════════════════════════════════════════

def report4(records, line):
    line('=' * 80)
    line('REPORT 4: GCN vs TRANSFORMER — ARCHITECTURAL COMPARISON')
    line('  Gap_arch_X = Transformer_X − GCN_X  (architecture gap on raw features)')
    line('  Gap_arch_U = Transformer_U − GCN_U  (architecture gap on spectral features)')
    line('  Positive = Transformer better. Negative = GCN better.')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, all 5 split seeds pooled.')
    line('NOTE: GCN uses graph adjacency (local). Transformer uses global self-attention.')
    line('      At ceiling (all labels given), which inductive bias is more effective?')
    line('=' * 80)

    line(f'\n{"Dataset":<22} {"base%":>5} | '
         f'{"GCN_X":>7} {"Trans_X":>8} {"Gap_X":>7} | '
         f'{"GCN_U":>7} {"Trans_U":>8} {"Gap_U":>7}')
    line('-' * 82)

    arch_gaps_x, arch_gaps_u = {}, {}
    for ds in ALL_DATASETS:
        recs = filt(records.get(ds,[]), optimizer=CANON_OPT, lr=CANON_LR)
        base = baseline(recs)
        if not recs:
            line(f'{ds:<22}  [no data]'); continue

        gcnx = mf(test_accs(recs,'GCN_X'))
        gcnu = mf(test_accs(recs,'GCN_U'))
        tx   = mf(test_accs(recs,'Transformer_X'))
        tu   = mf(test_accs(recs,'Transformer_U'))

        gap_x = tx - gcnx
        gap_u = tu - gcnu
        arch_gaps_x[ds] = gap_x
        arch_gaps_u[ds] = gap_u

        def f(v):  return f'{v:7.2f}' if not math.isnan(v) else '   skip'
        def gp(v): return f'{v:>+7.2f}' if not math.isnan(v) else '   skip'
        mx = ' ***' if not math.isnan(gap_x) and abs(gap_x) > 5 else ''
        mu = ' ***' if not math.isnan(gap_u) and abs(gap_u) > 5 else ''
        line(f'{ds:<22} {base:>5.1f} | '
             f'{f(gcnx)} {f(tx)} {gp(gap_x)}{mx} | '
             f'{f(gcnu)} {f(tu)} {gp(gap_u)}{mu}')

    line('')
    vx = [(ds,v) for ds,v in arch_gaps_x.items() if not math.isnan(v)]
    vu = [(ds,v) for ds,v in arch_gaps_u.items() if not math.isnan(v)]
    if vx:
        vals = [v for _,v in vx]
        pos  = [ds for ds,v in vx if v > 2]
        neg  = [ds for ds,v in vx if v < -2]
        line(f'  Gap_arch_X range: {min(vals):+.2f} to {max(vals):+.2f} pp')
        if pos: line(f'    Trans > GCN on X: {pos}')
        if neg: line(f'    GCN  > Trans on X: {neg}')
    if vu:
        vals = [v for _,v in vu]
        pos  = [ds for ds,v in vu if v > 2]
        neg  = [ds for ds,v in vu if v < -2]
        line(f'  Gap_arch_U range: {min(vals):+.2f} to {max(vals):+.2f} pp')
        if pos: line(f'    Trans > GCN on U: {pos}')
        if neg: line(f'    GCN  > Trans on U: {neg}')

    line('')
    line('KEY QUESTION: Does the spectral pre-processing (U) change WHICH architecture')
    line('wins? If Gap_arch_X and Gap_arch_U have opposite signs on any dataset,')
    line('that is a regime shift — U changes the relative advantage of the architectures.')
    line('')
    line(f'  {"Dataset":<22} {"Gap_arch_X":>12} {"Gap_arch_U":>12} {"Regime shift?":>14}')
    line('  ' + '-' * 65)
    for ds in ALL_DATASETS:
        gx = arch_gaps_x.get(ds, float('nan'))
        gu = arch_gaps_u.get(ds, float('nan'))
        if math.isnan(gx) or math.isnan(gu):
            shift = 'N/A (skipped)'
        elif (gx > 2 and gu < -2) or (gx < -2 and gu > 2):
            shift = 'YES — sign flip'
        elif (gx > 2) == (gu > 0) and abs(gu - gx) < 3:
            shift = 'no (consistent)'
        else:
            shift = 'partial'
        line(f'  {ds:<22} {fp(gx):>12} {fp(gu):>12} {shift:>14}')


# ══════════════════════════════════════════════════════════════════════════════
# R5 — Row-norm effect: GCN vs Transformer
# ══════════════════════════════════════════════════════════════════════════════

def report5(records, line):
    line('=' * 80)
    line('REPORT 5: ROW-NORM EFFECT — GCN vs TRANSFORMER')
    line('  Gap_RN_GCN   = GCN_rowNorm_U − GCN_U')
    line('  Gap_RN_Trans = Transformer_rowNorm_U − Transformer_U')
    line('  For GCN: row-norm at each layer INPUT (Option B).')
    line('  For Transformer: row-norm at input only (before first layer).')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, all 5 split seeds pooled.')
    line('=' * 80)

    line(f'\n{"Dataset":<22} {"base%":>5} | '
         f'{"GCN_U":>7} {"GCN_rnU":>8} {"RN_GCN":>8} | '
         f'{"T_U":>7} {"T_rnU":>8} {"RN_T":>8}')
    line('-' * 80)

    for ds in ALL_DATASETS:
        recs  = filt(records.get(ds,[]), optimizer=CANON_OPT, lr=CANON_LR)
        base  = baseline(recs)
        if not recs:
            line(f'{ds:<22}  [no data]'); continue

        gcnu  = mf(test_accs(recs, 'GCN_U'))
        gcnrn = mf(test_accs(recs, 'GCN_rowNorm_U'))
        tu    = mf(test_accs(recs, 'Transformer_U'))
        trn   = mf(test_accs(recs, 'Transformer_rowNorm_U'))
        rn_g  = gcnrn - gcnu
        rn_t  = trn   - tu

        def f(v):  return f'{v:7.2f}' if not math.isnan(v) else '   skip'
        def gp(v): return f'{v:>+8.2f}' if not math.isnan(v) else '    skip'
        mg = ' *' if not math.isnan(rn_g) and abs(rn_g) > 3 else ''
        mt = ' *' if not math.isnan(rn_t) and abs(rn_t) > 3 else ''
        line(f'{ds:<22} {base:>5.1f} | {f(gcnu)} {f(gcnrn)} {gp(rn_g)}{mg} | '
             f'{f(tu)} {f(trn)} {gp(rn_t)}{mt}')

    line('')
    line('* = |effect| > 3pp.  Positive = row-norm helps. Negative = row-norm hurts.')
    line('Sign consistency across GCN and Transformer reveals whether the effect')
    line('is tied to the spectral geometry or the aggregation mechanism.')


# ══════════════════════════════════════════════════════════════════════════════
# R6 — Optimizer sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def report6(records, line):
    line('=' * 80)
    line('REPORT 6: OPTIMIZER SENSITIVITY')
    line('Best LR per optimizer | Mean±std across all split × train seeds')
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
                    cols[opt] = ('   skip      ', float('nan')); continue
                r  = filt(recs, optimizer=opt, lr=blr)
                ta = test_accs(r, mk)
                if not ta:
                    cols[opt] = ('   skip      ', float('nan'))
                else:
                    cols[opt] = (ms(ta) + f'[lr={blr}]', mf(ta))
            gap   = cols['adam'][1] - cols['sgd'][1]
            gap_s = f'{gap:>+8.2f} pp' if not math.isnan(gap) else '   N/A    '
            line(f'  {ds:<22} {base:>5.1f} | {cols["sgd"][0]:>18} | '
                 f'{cols["adam"][0]:>18} | {gap_s:>15}')


# ══════════════════════════════════════════════════════════════════════════════
# R7 — Split seed stability
# ══════════════════════════════════════════════════════════════════════════════

def report7(records, line):
    line('=' * 80)
    line('REPORT 7: SPLIT SEED STABILITY')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}')
    line('split_std = std of per-seed means.  seed_std = mean of per-seed stds.')
    line('X_label_label and U_label_label are deterministic — variance = train partition.')
    line('=' * 80)

    for mk in MODEL_KEYS:
        line(f'\n  Model: {mk}')
        line(f'  {"Dataset":<22} '
             + ' '.join(f'{"ss"+str(s):>8}' for s in SPLIT_SEEDS)
             + f' | {"split_std":>10} {"seed_std":>10} | {"stable?":>8}')
        line('  ' + '-' * (22 + 9*len(SPLIT_SEEDS) + 35))
        for ds in ALL_DATASETS:
            recs = records.get(ds, [])
            seed_means, seed_stds, seed_strs = [], [], []
            for ss in SPLIT_SEEDS:
                r  = filt(recs, split_seed=ss, optimizer=CANON_OPT, lr=CANON_LR)
                ta = test_accs(r, mk)
                if ta:
                    seed_means.append(mf(ta)); seed_stds.append(sf(ta))
                    seed_strs.append(f'{mf(ta):6.2f}')
                else:
                    seed_strs.append('  skip')
            if len(seed_means) < 2:
                line(f'  {ds:<22}  [insufficient data / skipped]'); continue
            split_std = float(np.std(seed_means))
            seed_std  = float(np.mean(seed_stds))
            stable    = 'YES' if (seed_std > 0 and split_std < seed_std) else 'NO'
            vals_str  = ' '.join(f'{s:>8}' for s in seed_strs)
            line(f'  {ds:<22} {vals_str} | {split_std:>10.3f} {seed_std:>10.3f} | {stable:>8}')


# ══════════════════════════════════════════════════════════════════════════════
# R8 — Training curves
# ══════════════════════════════════════════════════════════════════════════════

def report8(records, line):
    line('=' * 80)
    line('REPORT 8: TRAINING CURVES — EPOCH CHECKPOINTS')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, all 5 split seeds pooled')
    line(f'Datasets shown: {CURVE_DATASETS}')
    line('=' * 80)
    for ds in CURVE_DATASETS:
        recs = filt(records.get(ds,[]), optimizer=CANON_OPT, lr=CANON_LR)
        base = baseline(recs)
        line(f'\n  Dataset: {ds}  (random baseline = {base:.2f}%)')
        line(f'  {"Model":<24}' + ''.join(f'  ep{ep:>3}' for ep in EPOCH_CHECKS))
        line('  ' + '-' * (24 + 8*len(EPOCH_CHECKS)))
        for mk in MODEL_KEYS:
            curves = test_curves(recs, mk)
            if not curves:
                line(f'  {mk:<24}  [no data / skipped]'); continue
            arr  = np.stack(curves).mean(axis=0)
            vals = ''.join(f'  {arr[ep-1]:>5.2f}' for ep in EPOCH_CHECKS)
            line(f'  {mk:<24}{vals}')


# ══════════════════════════════════════════════════════════════════════════════
# R9 — Verified findings
# ══════════════════════════════════════════════════════════════════════════════

def report_findings(records, line):
    line('=' * 80)
    line('REPORT 9: VERIFIED FINDINGS')
    line('Numbers first, conclusions second.')
    line(f'Canonical = {CANON_OPT}, lr={CANON_LR}, all 5 split seeds pooled.')
    line('Skipped Transformer entries excluded from all statistics.')
    line('=' * 80)

    canon = {}
    for ds in ALL_DATASETS:
        recs = filt(records.get(ds,[]), optimizer=CANON_OPT, lr=CANON_LR)
        canon[ds] = {mk: mf(test_accs(recs, mk)) for mk in MODEL_KEYS}
        canon[ds]['base'] = baseline(recs)

    # Finding 1: GCN ceiling
    line('')
    line('─' * 78)
    line('FINDING 1: GCN_X ceiling — transductive label propagation via GCN')
    line('─' * 78)
    line('')
    gcnx_vals = {}
    for ds in ALL_DATASETS:
        v = canon[ds]['GCN_X']
        gcnx_vals[ds] = v
        base = canon[ds]['base']
        ratio = v/base if not math.isnan(base) and base > 0 else float('nan')
        line(f'  {ds:<22}: {f2(v)}%  ({fp(ratio-1) if not math.isnan(ratio) else "N/A"}× above random)')
    valid = [v for v in gcnx_vals.values() if not math.isnan(v)]
    if valid: line(f'  Range: {min(valid):.2f}% to {max(valid):.2f}%')

    # Finding 2: Ceiling gap per architecture
    line('')
    line('─' * 78)
    line('FINDING 2: Ceiling gap Gap_U — GCN vs Transformer')
    line('─' * 78)
    line('span(X)=span(U) → gap is geometric optimization effect, not information.')
    line('')
    gap_gcn, gap_trans = {}, {}
    for ds in ALL_DATASETS:
        gcnx = canon[ds]['GCN_X']
        gcnu = canon[ds]['GCN_U']
        tx   = canon[ds]['Transformer_X']
        tu   = canon[ds]['Transformer_U']
        gap_gcn[ds]   = gcnu - gcnx
        gap_trans[ds] = tu   - tx
        mg = '*** ' if not math.isnan(gap_gcn[ds])   and abs(gap_gcn[ds])   > 5 else ''
        mt = '*** ' if not math.isnan(gap_trans[ds]) and abs(gap_trans[ds]) > 5 else ''
        line(f'  {ds:<22}  GCN: {fp(gap_gcn[ds]):>7} pp {mg} | '
             f'Transformer: {fp(gap_trans[ds]):>7} pp {mt}')

    vg = [v for v in gap_gcn.values()   if not math.isnan(v)]
    vt = [v for v in gap_trans.values() if not math.isnan(v)]
    if vg: line(f'  Gap_U GCN   range: {min(vg):+.2f} to {max(vg):+.2f} pp')
    if vt: line(f'  Gap_U Trans range: {min(vt):+.2f} to {max(vt):+.2f} pp')

    # Finding 3: Architecture comparison
    line('')
    line('─' * 78)
    line('FINDING 3: GCN vs Transformer — which benefits more from spectral pre-processing?')
    line('─' * 78)
    line('')
    same, different = [], []
    for ds in ALL_DATASETS:
        gg = gap_gcn.get(ds, float('nan'))
        gt = gap_trans.get(ds, float('nan'))
        if math.isnan(gg) or math.isnan(gt):
            continue
        if (gg > 2 and gt > 2) or (gg < -2 and gt < -2) or (abs(gg) <= 2 and abs(gt) <= 2):
            same.append(ds)
        else:
            different.append(f'{ds}(GCN={gg:+.1f},T={gt:+.1f})')
    if same:      line(f'  Same direction: {same}')
    if different: line(f'  Different response to U: {different}')

    # Summary table
    line('')
    line('─' * 78)
    line('SUMMARY — KEY NUMBERS  [VERIFIED from canonical config]')
    line('─' * 78)
    line(f'  {"Dataset":<22} | {"GCN_X":>8} {"GCN_U":>8} {"G_U_gcn":>8} | '
         f'{"T_X":>8} {"T_U":>8} {"G_U_t":>8}')
    line('  ' + '-' * 78)
    for ds in ALL_DATASETS:
        gx   = canon[ds]['GCN_X']
        gu   = canon[ds]['GCN_U']
        tx   = canon[ds]['Transformer_X']
        tu   = canon[ds]['Transformer_U']
        gg   = gap_gcn.get(ds, float('nan'))
        gt   = gap_trans.get(ds, float('nan'))
        line(f'  {ds:<22} | {f2(gx):>8} {f2(gu):>8} {fp(gg):>8} | '
             f'{f2(tx):>8} {f2(tu):>8} {fp(gt):>8}')
    line('')
    line('  SCIENTIFIC NOTE: Accuracy here measures label propagation, not generalisation.')
    line('  The key question is: does Gap_U differ between GCN and Transformer?')
    line('  If yes: the spectral pre-processing interacts with the inductive bias.')


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
    line('EXPERIMENT 2-v2: FULL LABEL INFORMATION + GNN vs TRANSFORMER — ANALYSIS')
    line('SpectralGeometryMisalignment / label_label_v2_experiment.py')
    line(f'Date: {date.today().strftime("%B %Y")}')
    line('=' * 80)
    line('')
    line('EXPERIMENTAL SETUP')
    line('------------------')
    line('X_label_label: ALL nodes → true one-hot label. Deterministic.')
    line('U_label_label: Rayleigh-Ritz of (L,D) on X_label_label.')
    line('span(X) = span(U) → all gaps are optimization geometry.')
    line('')
    line('*** Scientific probe — test labels in features. NOT predictive. ***')
    line('')
    line('6 models:')
    line('  GCN:         GCN_X | GCN_U (StandardScaler) | GCN_rowNorm_U (row-norm, Option B)')
    line('  Transformer: Transformer_X | Transformer_U | Transformer_rowNorm_U')
    line('  Transformer = 2-layer pre-norm TransformerEncoder, global self-attention,')
    line('  no graph adjacency, no positional encoding. hidden=256, heads=4, ff=512.')
    line('  Skipped for datasets with n > 25000 (O(N^2) infeasible).')
    line('')
    line('Training:  SGD + Adam, LR∈{0.001,0.01,0.1}, 500 epochs, WD=0')
    line('Seeds:     5 train × 5 random split seeds = 25 runs per (model, opt, lr)')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, all 5 split seeds pooled')
    line('')

    print('Loading results...')
    records = load_all()
    line('')

    completeness_check(records, line);  line('')
    dataset_metadata(records, line);    line('')
    report2(records, line);             line('')
    report1(records, line);             line('')
    report3(records, line);             line('')
    report4(records, line);             line('')
    report5(records, line);             line('')
    report6(records, line);             line('')
    report7(records, line);             line('')
    report8(records, line);             line('')
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
