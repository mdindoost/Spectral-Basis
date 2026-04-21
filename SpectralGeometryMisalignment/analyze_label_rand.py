"""
SpectralGeometryMisalignment/analyze_label_rand.py

Post-run analysis for label_rand_experiment.py.
Produces: results/label_rand/analysis/analysis_report.txt

EXPERIMENT: Partial label information.
  X_label_rand: train nodes → one-hot label, val/test → random N(0,1)
  U_label_rand: Rayleigh-Ritz of (L,D) on X_label_rand, num_classes eigenvectors

3 models: GCN_X (direct X), GCN_U (U + StandardScaler), GCN_rowNorm_U (U + row-norm)

Reports:
  META  Dataset metadata
  COMP  Completeness check
  R1    Main results table  — canonical config (adam lr=0.01, random ss=0)
  R2    Above-random check  — flag anything below random baseline
  R3    Central gap         — Gap_U = GCN_U − GCN_X (positive = spectral helps)
  R4    Row-norm effect     — Gap_RN = GCN_rowNorm_U − GCN_U
  R5    Optimizer sensitivity — SGD best-LR vs Adam best-LR
  R6    Fixed vs random split — split-type consistency
  R7    xrand draw stability  — variance across 3 draws
  R8    Training curves       — epoch checkpoints (selected datasets)
  R9    Verified findings

Canonical config: adam, lr=0.01, random split, split_seed=0
  → mean over xrand_seeds={0,1,2} × train_seeds={0..4} = up to 15 runs per cell

Best-LR selection: highest mean val_acc_curve[-1] across all seeds at that config.

Usage:
  /home/md724/Spectral-Basis/venv/bin/python \\
      SpectralGeometryMisalignment/analyze_label_rand.py
"""

import os, sys, json, math
from datetime import date
import numpy as np
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(_HERE, 'results', 'label_rand')
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
XRAND_SEEDS = [0, 1, 2]
SPLIT_SEEDS = [0, 1, 2, 3, 4]

CANON_OPT        = 'adam'
CANON_LR         = 0.01
CANON_SPLIT_TYPE = 'random'
CANON_SPLIT_SEED = 0

CURVE_DATASETS = ['cora', 'pubmed', 'ogbn_arxiv']
EPOCH_CHECKS   = [50, 100, 200, 300, 400, 500]

# Expected total records per dataset:
# 3 xrand × (1 fixed + 5 random) × 2 opts × 3 lrs × 5 train = 540
EXPECTED_RECORDS = 3 * 6 * 2 * 3 * 5


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

def filt(recs, split_type=None, split_seed=None, xrand_seed=None,
         optimizer=None, lr=None):
    out = recs
    if split_type  is not None: out = [r for r in out if r.get('split_type')  == split_type]
    if split_seed  is not None: out = [r for r in out if r.get('split_seed')  == split_seed]
    if xrand_seed  is not None: out = [r for r in out if r.get('xrand_seed')  == xrand_seed]
    if optimizer   is not None: out = [r for r in out if r.get('optimizer')   == optimizer]
    if lr          is not None: out = [r for r in out if abs(r.get('lr',0)-lr) < 1e-9]
    return out


# ── Extraction helpers ─────────────────────────────────────────────────────────

def test_accs(recs, mk):
    return [r[mk]['final_test_acc']*100 for r in recs
            if mk in r and not math.isnan(r[mk]['final_test_acc'])]

def val_accs(recs, mk):
    vals = []
    for r in recs:
        if mk in r:
            c = r[mk].get('val_acc_curve', [])
            if c and not math.isnan(c[-1]):
                vals.append(c[-1]*100)
    return vals

def test_curves(recs, mk):
    curves = []
    for r in recs:
        if mk in r:
            c = r[mk].get('test_acc_curve', [])
            if c and len(c)==500 and not any(math.isnan(v) for v in c):
                curves.append(np.array(c)*100)
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

def best_lr_for(recs_ds, mk, optimizer, split_type=CANON_SPLIT_TYPE):
    """Select LR with highest mean val_acc_curve[-1] across all seeds."""
    best_lr, best_mean = None, -1.0
    for lr in LRS:
        r = filt(recs_ds, split_type=split_type, optimizer=optimizer, lr=lr)
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
    line('(3 xrand_seeds × 6 splits × 2 opts × 3 lrs × 5 train_seeds = 540)')
    line('=' * 80)
    any_incomplete = False
    for ds in ALL_DATASETS:
        n = len(records.get(ds, []))
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
         f'{"d_eff":>6} {"base%":>6} {"n_train(rnd)":>13} {"records":>8}')
    line('-' * 90)
    for ds in ALL_DATASETS:
        recs = records.get(ds, [])
        if not recs:
            line(f'{ds:<22}  [no data]')
            continue
        r      = recs[0]
        n      = r.get('n_train',0) + r.get('n_val',0) + r.get('n_test',0)
        nc     = r.get('num_classes', 'N/A')
        idim   = r.get('input_dim', 'N/A')
        deff   = r.get('d_eff', 'N/A')
        base   = r.get('random_baseline', float('nan'))
        # n_train from a random split record
        rr = filt(recs, split_type='random')
        ntr = rr[0].get('n_train','N/A') if rr else 'N/A'
        line(f'{ds:<22} {n:>8,} {str(nc):>8} {str(idim):>10} {str(deff):>6} '
             f'{base:>6.2f} {str(ntr):>13} {len(recs):>8}')
    line('')
    line('input_dim = num_classes (X_label_rand has num_classes columns)')
    line('d_eff     = number of Rayleigh-Ritz eigenvectors (≤ num_classes)')
    line('n_train   = training nodes on a random 60/20/20 split')


# ══════════════════════════════════════════════════════════════════════════════
# R1 — Main results table
# ══════════════════════════════════════════════════════════════════════════════

def report1(records, line):
    line('=' * 80)
    line('REPORT 1: MAIN RESULTS TABLE')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, {CANON_SPLIT_TYPE} split, '
         f'split_seed={CANON_SPLIT_SEED}')
    line('Mean ± std over xrand_seeds={0,1,2} × train_seeds={0..4} (up to 15 runs)')
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
                r  = filt(recs, split_type=CANON_SPLIT_TYPE, optimizer=opt, lr=blr)
                ta = test_accs(r, mk)
                cols.append(ms(ta) + f' [lr={blr}]')
            line(f'  {ds:<22} {base:>5.1f} | {cols[0]:>20} | {cols[1]:>20} | {cols[2]:>20}')

    line('')
    line('Best LR selected by mean val_acc_curve[-1] across all xrand × train seeds.')


# ══════════════════════════════════════════════════════════════════════════════
# R2 — Above-random check
# ══════════════════════════════════════════════════════════════════════════════

def report2(records, line):
    line('=' * 80)
    line('REPORT 2: ABOVE-RANDOM CHECK')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, {CANON_SPLIT_TYPE} '
         f'split_seed={CANON_SPLIT_SEED}')
    line('Ratio = mean_test_acc / random_baseline  (>1.0 = above random)')
    line('ANOMALY if ratio < 1.0')
    line('=' * 80)

    hdr = (f'{"Dataset":<22} {"base%":>5} | '
           + '  '.join(f'{m:>15}' for m in MODEL_KEYS))
    line(hdr)
    line('-' * len(hdr))

    anomalies = []
    for ds in ALL_DATASETS:
        recs = filt(records.get(ds, []), split_type=CANON_SPLIT_TYPE,
                    split_seed=CANON_SPLIT_SEED, optimizer=CANON_OPT, lr=CANON_LR)
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


# ══════════════════════════════════════════════════════════════════════════════
# R3 — Central gap: Gap_U = GCN_U − GCN_X
# ══════════════════════════════════════════════════════════════════════════════

def report3(records, line):
    line('=' * 80)
    line('REPORT 3: CENTRAL GAP — Gap_U = GCN_U − GCN_X')
    line('Positive Gap_U = spectral projection (U_label_rand) helps GCN.')
    line('Negative Gap_U = direct features (X_label_rand) are better for GCN.')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, {CANON_SPLIT_TYPE} split_seed={CANON_SPLIT_SEED}')
    line('Mean over xrand_seeds × train_seeds.')
    line('=' * 80)

    line(f'\n{"Dataset":<22} {"base%":>5} | {"GCN_X":>7} {"GCN_U":>7} '
         f'{"Gap_U":>7} | {"GCN_rnU":>8} {"Gap_RN":>8}')
    line('-' * 72)

    for ds in ALL_DATASETS:
        recs = filt(records.get(ds, []), split_type=CANON_SPLIT_TYPE,
                    split_seed=CANON_SPLIT_SEED, optimizer=CANON_OPT, lr=CANON_LR)
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
    line('Gap_U  = GCN_U − GCN_X        (effect of spectral projection on GCN)')
    line('Gap_RN = GCN_rowNorm_U − GCN_U (additional effect of row-norm on spectral GCN)')
    line('NOTE: span(X_label_rand) = span(U_label_rand) → any gap is geometry, not info.')

    # Also show at all split seeds (random, adam, lr=0.01) pooled
    line('')
    line('── All random splits pooled (split_seeds 0–4, adam, lr=0.01) ──')
    line(f'{"Dataset":<22} {"base%":>5} | {"GCN_X":>7} {"GCN_U":>7} '
         f'{"Gap_U":>7} | {"GCN_rnU":>8} {"Gap_RN":>8}')
    line('-' * 72)
    for ds in ALL_DATASETS:
        recs = filt(records.get(ds, []), split_type='random',
                    optimizer=CANON_OPT, lr=CANON_LR)
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
        line(f'{ds:<22} {base:>5.1f} | {f(gcnx)} {f(gcnu)} {gap_u:>+7.2f} | '
             f'{f(gcnrn)} {gap_rn:>+8.2f}')


# ══════════════════════════════════════════════════════════════════════════════
# R4 — Optimizer sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def report4(records, line):
    line('=' * 80)
    line('REPORT 4: OPTIMIZER SENSITIVITY')
    line(f'{CANON_SPLIT_TYPE} split | Best LR per optimizer | '
         f'Mean±std across xrand_seeds × train_seeds')
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
                r  = filt(recs, split_type=CANON_SPLIT_TYPE, optimizer=opt, lr=blr)
                ta = test_accs(r, mk)
                cols[opt] = (ms(ta) + f'[lr={blr}]', mf(ta))
            gap = cols['adam'][1] - cols['sgd'][1]
            gap_s = f'{gap:>+8.2f} pp' if not math.isnan(gap) else '   N/A    '
            line(f'  {ds:<22} {base:>5.1f} | {cols["sgd"][0]:>18} | '
                 f'{cols["adam"][0]:>18} | {gap_s:>15}')


# ══════════════════════════════════════════════════════════════════════════════
# R5 — Fixed vs random split
# ══════════════════════════════════════════════════════════════════════════════

def report5(records, line):
    line('=' * 80)
    line('REPORT 5: FIXED vs RANDOM SPLIT CONSISTENCY')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}')
    line('Fixed:  split_seed=0, mean over xrand × train seeds')
    line('Random: mean over split_seeds={0..4} × xrand × train seeds')
    line('NOTE: X_label_rand is split-dependent — different splits → different one-hot nodes.')
    line('      A large Δ here means the label-seeding pattern matters, not just split size.')
    line('=' * 80)

    for mk in MODEL_KEYS:
        line(f'\n  Model: {mk}')
        line(f'  {"Dataset":<22} {"base%":>5} | {"Fixed":>11} | {"Random":>11} | {"Δ (rnd−fix)":>12}')
        line('  ' + '-' * 65)
        for ds in ALL_DATASETS:
            recs = records.get(ds, [])
            base = baseline(recs)
            fix_recs = filt(recs, split_type='fixed', split_seed=0,
                            optimizer=CANON_OPT, lr=CANON_LR)
            rnd_recs = filt(recs, split_type='random',
                            optimizer=CANON_OPT, lr=CANON_LR)
            fix_m = mf(test_accs(fix_recs, mk))
            rnd_m = mf(test_accs(rnd_recs, mk))
            delta = rnd_m - fix_m
            fix_s = f'{fix_m:6.2f}%' if not math.isnan(fix_m) else '   N/A '
            rnd_s = f'{rnd_m:6.2f}%' if not math.isnan(rnd_m) else '   N/A '
            delta_s = f'{delta:>+8.2f} pp' if not math.isnan(delta) else '   N/A   '
            line(f'  {ds:<22} {base:>5.1f} | {fix_s:>11} | {rnd_s:>11} | {delta_s:>12}')

    line('')
    line('Large |Δ| = results strongly depend on which nodes become training (one-hot) nodes.')


# ══════════════════════════════════════════════════════════════════════════════
# R6 — xrand draw stability
# ══════════════════════════════════════════════════════════════════════════════

def report6(records, line):
    line('=' * 80)
    line('REPORT 6: XRAND DRAW STABILITY')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, {CANON_SPLIT_TYPE} '
         f'split_seed={CANON_SPLIT_SEED}')
    line('Per-draw mean computed over train_seeds={0..4}.')
    line('draw_std = std across 3 draws. seed_std = mean of per-draw stds.')
    line('Stable: draw_std << seed_std.')
    line('=' * 80)

    for mk in MODEL_KEYS:
        line(f'\n  Model: {mk}')
        line(f'  {"Dataset":<22} {"draw0":>8} {"draw1":>8} {"draw2":>8} | '
             f'{"draw_std":>9} {"seed_std":>9} | {"stable?":>8}')
        line('  ' + '-' * 72)
        for ds in ALL_DATASETS:
            recs = records.get(ds, [])
            draw_means, draw_stds, draw_strs = [], [], []
            for xs in XRAND_SEEDS:
                r  = filt(recs, split_type=CANON_SPLIT_TYPE,
                          split_seed=CANON_SPLIT_SEED,
                          optimizer=CANON_OPT, lr=CANON_LR, xrand_seed=xs)
                ta = test_accs(r, mk)
                if ta:
                    draw_means.append(mf(ta))
                    draw_stds.append(sf(ta))
                    draw_strs.append(f'{mf(ta):6.2f}')
                else:
                    draw_strs.append('   N/A')

            if len(draw_means) < 2:
                line(f'  {ds:<22}  [insufficient data]')
                continue

            draw_std = float(np.std(draw_means))
            seed_std = float(np.mean(draw_stds))
            stable   = 'YES' if (seed_std > 0 and draw_std < seed_std) else 'NO'
            line(f'  {ds:<22} {draw_strs[0]:>8} {draw_strs[1]:>8} {draw_strs[2]:>8} | '
                 f'{draw_std:>9.3f} {seed_std:>9.3f} | {stable:>8}')


# ══════════════════════════════════════════════════════════════════════════════
# R7 — Training curves
# ══════════════════════════════════════════════════════════════════════════════

def report7(records, line):
    line('=' * 80)
    line('REPORT 7: TRAINING CURVES — EPOCH CHECKPOINTS')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, {CANON_SPLIT_TYPE} '
         f'split_seed={CANON_SPLIT_SEED}')
    line('Values = mean test accuracy (%) across xrand_seeds × train_seeds.')
    line(f'Datasets shown: {CURVE_DATASETS}')
    line('=' * 80)

    for ds in CURVE_DATASETS:
        recs = filt(records.get(ds, []), split_type=CANON_SPLIT_TYPE,
                    split_seed=CANON_SPLIT_SEED, optimizer=CANON_OPT, lr=CANON_LR)
        base = baseline(recs)
        line(f'\n  Dataset: {ds}  (random baseline = {base:.2f}%)')
        line(f'  {"Model":<16}' + ''.join(f'  ep{ep:>3}' for ep in EPOCH_CHECKS))
        line('  ' + '-' * (16 + 8*len(EPOCH_CHECKS)))
        for mk in MODEL_KEYS:
            curves = test_curves(recs, mk)
            if not curves:
                line(f'  {mk:<16}  [no data]')
                continue
            arr  = np.stack(curves).mean(axis=0)
            vals = ''.join(f'  {arr[ep-1]:>5.2f}' for ep in EPOCH_CHECKS)
            line(f'  {mk:<16}{vals}')


# ══════════════════════════════════════════════════════════════════════════════
# R9 — Verified findings
# ══════════════════════════════════════════════════════════════════════════════

def report_findings(records, line):
    line('=' * 80)
    line('REPORT 9: VERIFIED FINDINGS')
    line('Numbers first, conclusions second.')
    line(f'Canonical = {CANON_OPT}, lr={CANON_LR}, {CANON_SPLIT_TYPE} split_seed={CANON_SPLIT_SEED}.')
    line('=' * 80)

    def f2(v): return f'{v:.2f}' if not math.isnan(v) else 'N/A'

    # Pre-compute canonical means
    canon = {}
    for ds in ALL_DATASETS:
        recs = filt(records.get(ds,[]), split_type=CANON_SPLIT_TYPE,
                    split_seed=CANON_SPLIT_SEED, optimizer=CANON_OPT, lr=CANON_LR)
        canon[ds] = {mk: mf(test_accs(recs, mk)) for mk in MODEL_KEYS}
        canon[ds]['base'] = baseline(recs)

    # Pre-compute all-random-splits means
    all_rnd = {}
    for ds in ALL_DATASETS:
        recs = filt(records.get(ds,[]), split_type='random',
                    optimizer=CANON_OPT, lr=CANON_LR)
        all_rnd[ds] = {mk: mf(test_accs(recs, mk)) for mk in MODEL_KEYS}
        all_rnd[ds]['base'] = baseline(recs)

    # Finding 1: Gap_U sign and magnitude
    line('')
    line('─' * 78)
    line('FINDING 1: Does spectral projection (U_label_rand) help or hurt GCN?')
    line('─' * 78)
    line('Gap_U = GCN_U − GCN_X at canonical config.')
    line('')
    gap_u_vals = {}
    for ds in ALL_DATASETS:
        gcnx = canon[ds]['GCN_X']
        gcnu = canon[ds]['GCN_U']
        gap  = gcnu - gcnx
        gap_u_vals[ds] = gap
        marker = ' *** positive (U helps)' if gap > 2 else (' *** negative (X better)' if gap < -2 else ' (near-zero)')
        line(f'  {ds:<22}: GCN_X={f2(gcnx)}%  GCN_U={f2(gcnu)}%  Gap_U={gap:>+.2f} pp{marker}')

    valid_gaps = [v for v in gap_u_vals.values() if not math.isnan(v)]
    pos  = [ds for ds,v in gap_u_vals.items() if v > 2]
    neg  = [ds for ds,v in gap_u_vals.items() if v < -2]
    line('')
    if valid_gaps:
        line(f'  Range: {min(valid_gaps):+.2f} to {max(valid_gaps):+.2f} pp')
        if pos:  line(f'  U helps  (Gap_U > +2pp): {pos}')
        if neg:  line(f'  X better (Gap_U < −2pp): {neg}')

    # Finding 2: Row-norm effect
    line('')
    line('─' * 78)
    line('FINDING 2: Effect of row-norm on U_label_rand (Gap_RN = GCN_rowNorm_U − GCN_U)')
    line('─' * 78)
    line('')
    gap_rn_vals = {}
    for ds in ALL_DATASETS:
        gcnu  = canon[ds]['GCN_U']
        gcnrn = canon[ds]['GCN_rowNorm_U']
        gap   = gcnrn - gcnu
        gap_rn_vals[ds] = gap
        marker = ' *** (helps)' if gap > 2 else (' *** (hurts)' if gap < -2 else '')
        line(f'  {ds:<22}: GCN_U={f2(gcnu)}%  GCN_rowNorm_U={f2(gcnrn)}%  Gap_RN={gap:>+.2f} pp{marker}')

    valid_rn = [v for v in gap_rn_vals.values() if not math.isnan(v)]
    if valid_rn:
        line(f'  Range: {min(valid_rn):+.2f} to {max(valid_rn):+.2f} pp')

    # Finding 3: Fixed vs random split effect on Gap_U
    line('')
    line('─' * 78)
    line('FINDING 3: Fixed vs random split — does the split size change Gap_U?')
    line('─' * 78)
    line('')
    line(f'  {"Dataset":<22} {"Gap_U fixed":>12} {"Gap_U random":>13} {"Δ":>8}')
    line('  ' + '-' * 58)
    for ds in ALL_DATASETS:
        fix_recs = filt(records.get(ds,[]), split_type='fixed', split_seed=0,
                        optimizer=CANON_OPT, lr=CANON_LR)
        fix_gap  = mf(test_accs(fix_recs,'GCN_U')) - mf(test_accs(fix_recs,'GCN_X'))
        rnd_gap  = all_rnd[ds]['GCN_U'] - all_rnd[ds]['GCN_X']
        delta    = rnd_gap - fix_gap
        def fp(v): return f'{v:>+.2f}' if not math.isnan(v) else '  N/A '
        line(f'  {ds:<22} {fp(fix_gap):>12} {fp(rnd_gap):>13} {fp(delta):>8}')

    # Summary
    line('')
    line('─' * 78)
    line('SUMMARY — KEY NUMBERS')
    line('─' * 78)
    vg = [v for v in gap_u_vals.values() if not math.isnan(v)]
    vr = [v for v in gap_rn_vals.values() if not math.isnan(v)]
    if vg: line(f'  Gap_U  range (canonical): {min(vg):+.2f} to {max(vg):+.2f} pp')
    if vr: line(f'  Gap_RN range (canonical): {min(vr):+.2f} to {max(vr):+.2f} pp')
    line('  All numbers [VERIFIED] from canonical config (adam lr=0.01, random ss=0).')


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
    line('EXPERIMENT 1: PARTIAL LABEL INFORMATION — FULL ANALYSIS REPORT')
    line('SpectralGeometryMisalignment / label_rand_experiment.py')
    line(f'Date: {date.today().strftime("%B %Y")}')
    line('=' * 80)
    line('')
    line('EXPERIMENTAL SETUP')
    line('------------------')
    line('X_label_rand: train nodes → one-hot class label, val/test → N(0,1).')
    line('U_label_rand: Rayleigh-Ritz of (L,D) on X_label_rand. d_eff ≤ num_classes.')
    line('span(X) = span(U) — gap between GCN(X) and GCN(U) is optimization geometry.')
    line('')
    line('3 models:  GCN_X | GCN_U (StandardScaler) | GCN_rowNorm_U (row-norm at layer input)')
    line('Training:  SGD + Adam, LR∈{0.001,0.01,0.1}, 500 epochs, WD=0')
    line('Seeds:     5 train × 3 xrand draws × 6 split configs (1 fixed + 5 random)')
    line(f'Canonical: {CANON_OPT}, lr={CANON_LR}, {CANON_SPLIT_TYPE} split, '
         f'split_seed={CANON_SPLIT_SEED}')
    line('')

    print('Loading results...')
    records = load_all()
    line('')

    completeness_check(records, line);  line('')
    dataset_metadata(records, line);    line('')
    report2(records, line);             line('')  # above-random (sanity gate)
    report1(records, line);             line('')  # main table
    report3(records, line);             line('')  # central gap
    report4(records, line);             line('')  # optimizer sensitivity
    report5(records, line);             line('')  # fixed vs random
    report6(records, line);             line('')  # draw stability
    report7(records, line);             line('')  # training curves
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
