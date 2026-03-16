"""
Exp 9 Analysis — Parts 2, 3, 4
================================
Run AFTER exp9_softmax_convergence.py has completed for all 9 datasets.

Loads:
  PAPER_RESULTS/{dataset}_fixed_lcc/k{k}/softmax/exp9_adam.json  (exp9 results)
  PAPER_RESULTS/{dataset}_fixed_lcc/k{k}/softmax/exp9_sgd.json
  PAPER_RESULTS/{dataset}_fixed_lcc/k{k}/metrics/results.json    (MLP Part A)
  PAPER_RESULTS/{dataset}_fixed_lcc/analytics/exp2_spectral_analysis_k{k}.json  (d_eff)

Computes (Part 2):
  Softmax_Part_A  = Test%(X_diff+Adam) - Test%(Y+Adam)
  MLP_Part_A      = sgc_mlp_acc - restricted_std_acc  [from existing results.json]
  Crossover_match = sign(Softmax_Part_A) == sign(MLP_Part_A) at each k
  SGD_vs_Adam_gap = Test%(Y+Adam) - Test%(Y+SGD)

Flags (Part 3):
  FLAG A: Softmax_Part_A < 5pp on monotonic dataset at any k
  FLAG B: Softmax_Part_A < 0 on monotonic dataset at any k
  FLAG C: Softmax crossover k ≠ MLP crossover k for the same dataset
  FLAG D: SGD_vs_Adam_gap on Y > 20pp at any dataset × k

Report (Part 4): 7 sections, numbers first then interpretation.

Usage:
  cd /home/md724/Spectral-Basis
  venv/bin/python PAPER_EXPERIMENTS/exp9_analysis.py
"""

import os
import json
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

DATASETS = [
    'cora', 'citeseer', 'pubmed',
    'ogbn-arxiv', 'wikics',
    'amazon-computers', 'amazon-photo',
    'coauthor-cs', 'coauthor-physics',
]
K_VALUES = [1, 2, 4, 6, 8, 10, 12, 20, 30]

# Behavioral classification from the main paper
MONOTONIC   = {'cora', 'citeseer', 'pubmed', 'coauthor-cs', 'coauthor-physics'}
CROSSOVER   = {'amazon-computers', 'amazon-photo', 'ogbn-arxiv', 'wikics'}

RESULTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PAPER_RESULTS')

# ============================================================================
# Data loading helpers
# ============================================================================

def load_exp9(dataset, k, opt):
    """Load exp9 JSON. Returns None if file missing."""
    path = os.path.join(RESULTS_BASE, f'{dataset}_fixed_lcc',
                        f'k{k}', 'softmax', f'exp9_{opt}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_mlp_results(dataset, k):
    """Load main paper results.json. Returns framework_analysis dict or None."""
    path = os.path.join(RESULTS_BASE, f'{dataset}_fixed_lcc',
                        f'k{k}', 'metrics', 'results.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get('framework_analysis', None)

def load_d_eff(dataset, k):
    """Load d_effective from exp2 analytics JSON. Returns int or None."""
    path = os.path.join(RESULTS_BASE, f'{dataset}_fixed_lcc',
                        'analytics', f'exp2_spectral_analysis_k{k}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get('d_effective', None)

# ============================================================================
# Build full data table
# Rows: dataset × k
# Columns: all metrics needed for Parts 2–4
# ============================================================================

print('Loading results...')

rows = []          # list of dicts, one per (dataset, k)
missing = []       # track missing files

for ds in DATASETS:
    for k in K_VALUES:
        adam = load_exp9(ds, k, 'adam')
        sgd  = load_exp9(ds, k, 'sgd')
        mlp  = load_mlp_results(ds, k)
        d_eff = load_d_eff(ds, k)

        if adam is None or sgd is None:
            missing.append(f'{ds} k={k}')
            continue

        # Accuracy values (as percentages)
        xd_adam = adam['experiments']['X_diff']['test_acc_mean'] * 100
        y_adam  = adam['experiments']['Y']['test_acc_mean']      * 100
        xd_sgd  = sgd['experiments']['X_diff']['test_acc_mean']  * 100
        y_sgd   = sgd['experiments']['Y']['test_acc_mean']       * 100

        xd_adam_std = adam['experiments']['X_diff']['test_acc_std'] * 100
        y_adam_std  = adam['experiments']['Y']['test_acc_std']      * 100

        # Convergence speed (90%, 95% of own peak)
        ep90_xd = adam['experiments']['X_diff'].get('epoch_to_90pct_peak_mean')
        ep90_y  = adam['experiments']['Y'].get('epoch_to_90pct_peak_mean')
        ep95_xd = adam['experiments']['X_diff'].get('epoch_to_95pct_peak_mean')
        ep95_y  = adam['experiments']['Y'].get('epoch_to_95pct_peak_mean')

        ep90_xd_sgd = sgd['experiments']['X_diff'].get('epoch_to_90pct_peak_mean')
        ep90_y_sgd  = sgd['experiments']['Y'].get('epoch_to_90pct_peak_mean')

        # Derived metrics
        softmax_part_a   = xd_adam - y_adam          # analogous to MLP Part A
        sgd_vs_adam_y    = y_adam - y_sgd             # how much Adam helps Y
        sgd_vs_adam_xd   = xd_adam - xd_sgd

        mlp_part_a = None
        if mlp:
            mlp_part_a = mlp.get('part_a_pp')        # SGC+MLP - Restricted+Std

        row = {
            'dataset':         ds,
            'k':               k,
            'd_eff':           d_eff,
            'behavioral_type': 'monotonic' if ds in MONOTONIC else 'crossover',
            # Adam
            'xd_adam':         xd_adam,
            'y_adam':          y_adam,
            'xd_adam_std':     xd_adam_std,
            'y_adam_std':      y_adam_std,
            # SGD
            'xd_sgd':          xd_sgd,
            'y_sgd':           y_sgd,
            # Derived
            'softmax_part_a':  softmax_part_a,
            'mlp_part_a':      mlp_part_a,
            'sgd_vs_adam_y':   sgd_vs_adam_y,
            'sgd_vs_adam_xd':  sgd_vs_adam_xd,
            'sign_match':      (None if mlp_part_a is None
                                else (softmax_part_a >= 0) == (mlp_part_a >= 0)),
            # Convergence speed (Adam only)
            'ep90_xd_adam':    ep90_xd,
            'ep90_y_adam':     ep90_y,
            'ep95_xd_adam':    ep95_xd,
            'ep95_y_adam':     ep95_y,
            # Convergence speed (SGD)
            'ep90_xd_sgd':     ep90_xd_sgd,
            'ep90_y_sgd':      ep90_y_sgd,
        }
        rows.append(row)

if missing:
    print(f'\n*** MISSING FILES ({len(missing)} configurations):')
    for m in missing:
        print(f'    {m}')
    if len(missing) == len(DATASETS) * len(K_VALUES):
        print('\nNo results found. Run exp9_softmax_convergence.py for all datasets first.')
        raise SystemExit(1)

print(f'Loaded {len(rows)} (dataset × k) configurations.')

# ============================================================================
# Helper: find crossover k for a dataset
# Crossover = first k where Softmax_Part_A changes sign (positive → negative)
# ============================================================================

def find_crossover(ds_rows, metric_key):
    """
    Returns the k where metric_key first goes negative (crossover), or None.
    ds_rows must be sorted by k.
    """
    prev_sign = None
    for r in ds_rows:
        val = r.get(metric_key)
        if val is None:
            continue
        sign = val >= 0
        if prev_sign is not None and sign != prev_sign and not sign:
            return r['k']
        prev_sign = sign
    return None

# Per-dataset row lookup
def ds_rows(ds):
    return sorted([r for r in rows if r['dataset'] == ds], key=lambda r: r['k'])

# ============================================================================
# PART 3 — Flags
# Evaluate before report so flags can be cited in the text
# ============================================================================

flags = []  # list of (flag_id, dataset, k, message, numbers)

for ds in DATASETS:
    drows = ds_rows(ds)
    btype = 'monotonic' if ds in MONOTONIC else 'crossover'

    softmax_co_k = find_crossover(drows, 'softmax_part_a')
    mlp_co_k     = find_crossover(drows, 'mlp_part_a')

    for r in drows:
        k  = r['k']
        sa = r['softmax_part_a']
        ma = r['mlp_part_a']
        gy = r['sgd_vs_adam_y']

        # FLAG A: Softmax_Part_A < 5pp on monotonic dataset at any k
        if btype == 'monotonic' and sa < 5.0:
            flags.append(('A', ds, k,
                f'Softmax_Part_A={sa:.2f}pp < 5pp on monotonic dataset',
                {'softmax_part_a': sa, 'xd_adam': r['xd_adam'], 'y_adam': r['y_adam']}))

        # FLAG B: Softmax_Part_A < 0 on monotonic dataset at any k
        if btype == 'monotonic' and sa < 0.0:
            flags.append(('B', ds, k,
                f'Softmax_Part_A={sa:.2f}pp < 0 — Y+Adam OUTPERFORMS X_diff+Adam',
                {'softmax_part_a': sa, 'xd_adam': r['xd_adam'], 'y_adam': r['y_adam']}))

        # FLAG D: SGD_vs_Adam_gap on Y > 20pp
        if gy > 20.0:
            flags.append(('D', ds, k,
                f'SGD_vs_Adam_gap on Y = {gy:.2f}pp > 20pp',
                {'y_adam': r['y_adam'], 'y_sgd': r['y_sgd'], 'gap': gy}))

    # FLAG C: Crossover mismatch between softmax and MLP
    # Only relevant for crossover datasets; for monotonic, both should be None
    if softmax_co_k != mlp_co_k:
        flags.append(('C', ds, None,
            f'Crossover k mismatch: softmax crossover at k={softmax_co_k}, '
            f'MLP crossover at k={mlp_co_k}',
            {'softmax_crossover_k': softmax_co_k, 'mlp_crossover_k': mlp_co_k}))

# ============================================================================
# PART 4 — Report
# ============================================================================

SEP  = '=' * 78
SEP2 = '-' * 78

def fmt(v, decimals=2):
    if v is None: return 'N/A'
    return f'{v:.{decimals}f}'

def fmt_ep(v):
    if v is None: return 'N/A'
    return f'{v:.1f}'

print('\n\n' + SEP)
print('EXP 9 ANALYSIS REPORT — Softmax Regression Convergence')
print('Datasets: all 9  |  k sweep: 1,2,4,6,8,10,12,20,30  |  Seeds: 15  |  Epochs: 500')
print(SEP)

# ────────────────────────────────────────────────────────────────────────────
# Section 1 — Overview table
# ────────────────────────────────────────────────────────────────────────────
print('\n' + SEP)
print('SECTION 1 — OVERVIEW TABLE')
print('Columns: Softmax_Part_A = X_diff+Adam% - Y+Adam%  (analogous to MLP Part A)')
print('         Crossover k = first k where Softmax_Part_A turns negative')
print(SEP)

hdr = (f'{"Dataset":<22} {"Type":<10} '
       f'{"SoftA k=1":>10} {"SoftA k=10":>11} {"SoftA k=30":>11} '
       f'{"Soft-CO k":>10} {"MLP-CO k":>9} {"CO match":>9}')
print(hdr)
print(SEP2)

for ds in DATASETS:
    drows_ds = ds_rows(ds)
    btype    = 'monotonic' if ds in MONOTONIC else 'crossover'

    sa_by_k = {r['k']: r['softmax_part_a'] for r in drows_ds}
    soft_co  = find_crossover(drows_ds, 'softmax_part_a')
    mlp_co   = find_crossover(drows_ds, 'mlp_part_a')
    co_match = ('yes' if soft_co == mlp_co
                else 'NO' if soft_co != mlp_co else '?')

    print(f'{ds:<22} {btype:<10} '
          f'{fmt(sa_by_k.get(1)):>10} {fmt(sa_by_k.get(10)):>11} '
          f'{fmt(sa_by_k.get(30)):>11} '
          f'{str(soft_co) if soft_co else "none":>10} '
          f'{str(mlp_co)  if mlp_co  else "none":>9} '
          f'{co_match:>9}')

print(SEP2)

# ────────────────────────────────────────────────────────────────────────────
# Section 2 — Full results tables, one per dataset
# ────────────────────────────────────────────────────────────────────────────
print('\n' + SEP)
print('SECTION 2 — FULL RESULTS TABLES  (all k values per dataset)')
print('Test accuracies in %.  Softmax_Part_A = X_diff+Adam - Y+Adam.')
print('MLP_Part_A = SGC+MLP - Restricted+Std  [from existing results.json].')
print(SEP)

for ds in DATASETS:
    drows_ds = ds_rows(ds)
    print(f'\n  Dataset: {ds.upper()}')
    print(f'  {"k":>4}  {"d_eff":>6}  '
          f'{"XD+SGD%":>9}  {"Y+SGD%":>8}  '
          f'{"XD+Adam%":>10}  {"Y+Adam%":>9}  '
          f'{"Soft_PartA":>11}  {"MLP_PartA":>10}')
    print(f'  {"-"*84}')
    for r in drows_ds:
        sign_warn = ' !' if r['mlp_part_a'] is not None and \
                            (r['softmax_part_a'] >= 0) != (r['mlp_part_a'] >= 0) else ''
        print(f'  {r["k"]:>4}  {str(r["d_eff"]) if r["d_eff"] else "?":>6}  '
              f'{fmt(r["xd_sgd"]):>9}  {fmt(r["y_sgd"]):>8}  '
              f'{fmt(r["xd_adam"]):>10}  {fmt(r["y_adam"]):>9}  '
              f'{fmt(r["softmax_part_a"]):>11}  '
              f'{fmt(r["mlp_part_a"]) if r["mlp_part_a"] is not None else "N/A":>10}'
              f'{sign_warn}')
    print()

# ────────────────────────────────────────────────────────────────────────────
# Section 3 — Crossover analysis
# ────────────────────────────────────────────────────────────────────────────
print('\n' + SEP)
print('SECTION 3 — CROSSOVER ANALYSIS')
print('Crossover datasets: amazon-computers, amazon-photo, ogbn-arxiv, wikics')
print('Question: does softmax crossover occur? At same k as MLP? Earlier or later?')
print(SEP)

for ds in CROSSOVER:
    drows_ds  = ds_rows(ds)
    soft_co   = find_crossover(drows_ds, 'softmax_part_a')
    mlp_co    = find_crossover(drows_ds, 'mlp_part_a')

    print(f'\n  {ds.upper()}')
    print(f'  {"k":>4}  {"Soft_PartA":>12}  {"MLP_PartA":>11}  {"sign_match":>11}')
    print(f'  {"-"*44}')
    for r in drows_ds:
        sm = 'yes' if r['sign_match'] else ('NO' if r['sign_match'] is False else '?')
        print(f'  {r["k"]:>4}  {fmt(r["softmax_part_a"]):>12}  '
              f'{fmt(r["mlp_part_a"]) if r["mlp_part_a"] is not None else "N/A":>11}  '
              f'{sm:>11}')

    print(f'\n  Softmax crossover k: {soft_co if soft_co else "none"}')
    print(f'  MLP crossover k:     {mlp_co  if mlp_co  else "none"}')

    if soft_co is None and mlp_co is None:
        print('  Interpretation: neither shows a crossover — consistent.')
    elif soft_co == mlp_co:
        print('  Interpretation: crossovers match — softmax and MLP flip at the same k.')
        print('  This implies the crossover is driven by the feature geometry (X_diff vs U),')
        print('  not by the MLP nonlinearity or depth.')
    elif soft_co is not None and mlp_co is None:
        print(f'  Interpretation: softmax crosses over at k={soft_co} but MLP does not.')
        print('  MLP nonlinearity may be compensating for the Y advantage at high k.')
    elif soft_co is None and mlp_co is not None:
        print(f'  Interpretation: MLP crosses over at k={mlp_co} but softmax does not.')
        print('  The crossover requires MLP nonlinearity to manifest — '
              'linear softmax on Y cannot exploit the high-k spectral structure.')
    else:
        direction = 'earlier' if soft_co < mlp_co else 'later'
        print(f'  Interpretation: softmax crosses at k={soft_co}, MLP at k={mlp_co} '
              f'({direction} for softmax).')
        if soft_co < mlp_co:
            print('  Y becomes better than X_diff for softmax before MLP — '
                  'the linear classifier exploits the spectral advantage sooner.')
        else:
            print('  MLP exploits the high-k spectral advantage before softmax does — '
                  'nonlinearity accelerates the crossover.')

# ────────────────────────────────────────────────────────────────────────────
# Section 4 — SGD vs Adam comparison
# ────────────────────────────────────────────────────────────────────────────
print('\n\n' + SEP)
print('SECTION 4 — SGD vs ADAM COMPARISON ON Y')
print('SGD_vs_Adam_gap_Y = Test%(Y+Adam) - Test%(Y+SGD)  (positive = Adam helps Y)')
print('Also shown: same gap on X_diff for reference.')
print(SEP)

print(f'\n  {"Dataset":<22}  '
      f'{"k=1 gap Y":>10}  {"k=1 gap XD":>11}  '
      f'{"k=10 gap Y":>11}  {"k=10 gap XD":>12}')
print(f'  {"-"*72}')
for ds in DATASETS:
    drows_ds = ds_rows(ds)
    by_k = {r['k']: r for r in drows_ds}
    r1  = by_k.get(1)
    r10 = by_k.get(10)
    g1_y  = fmt(r1['sgd_vs_adam_y'])   if r1  else 'N/A'
    g1_x  = fmt(r1['sgd_vs_adam_xd'])  if r1  else 'N/A'
    g10_y = fmt(r10['sgd_vs_adam_y'])  if r10 else 'N/A'
    g10_x = fmt(r10['sgd_vs_adam_xd']) if r10 else 'N/A'
    print(f'  {ds:<22}  {g1_y:>10}  {g1_x:>11}  {g10_y:>11}  {g10_x:>12}')

print()
print('  Interpretation notes:')
print('  - Positive gap = Adam helps relative to SGD.')
print('  - If gap_Y >> gap_XD at same k: Adam specifically helps Y (interesting).')
print('  - If gap_Y < gap_XD consistently: Adam helps X_diff MORE (widening gap).')
print('  - From pilot: Cora k=1 — gap_Y≈+4.8pp, gap_XD≈+8.7pp (Adam widens gap).')

# ────────────────────────────────────────────────────────────────────────────
# Section 5 — Rank collapse at k=20, k=30
# ────────────────────────────────────────────────────────────────────────────
print('\n' + SEP)
print('SECTION 5 — RANK COLLAPSE AND SOFTMAX PERFORMANCE AT k=20, k=30')
print('Datasets where d_eff drops at high k: cora, citeseer, coauthor-cs/physics, amazon-photo')
print('Question: does reduced d_eff at k=30 shrink Softmax_Part_A vs k=10?')
print(SEP)

rank_collapse_datasets = ['cora', 'citeseer', 'coauthor-cs', 'coauthor-physics', 'amazon-photo']
compare_ks = [10, 20, 30]

print(f'\n  {"Dataset":<22}  ', end='')
for k in compare_ks:
    print(f'{"d_eff k="+str(k):>12}  {"SoftA k="+str(k):>12}  ', end='')
print()
print(f'  {"-"*90}')

for ds in rank_collapse_datasets:
    drows_ds = ds_rows(ds)
    by_k = {r['k']: r for r in drows_ds}
    print(f'  {ds:<22}  ', end='')
    for k in compare_ks:
        r = by_k.get(k)
        if r:
            print(f'{str(r["d_eff"]) if r["d_eff"] else "?":>12}  '
                  f'{fmt(r["softmax_part_a"]):>12}  ', end='')
        else:
            print(f'{"N/A":>12}  {"N/A":>12}  ', end='')
    print()

print()
print('  If Softmax_Part_A shrinks at k=30 relative to k=10 on rank-collapse datasets,')
print('  this supports the hypothesis that reduced d_eff reduces the feature-label')
print('  misalignment problem (fewer eigenvectors → class signal more concentrated).')

# ────────────────────────────────────────────────────────────────────────────
# Section 6 — Flags
# ────────────────────────────────────────────────────────────────────────────
print('\n' + SEP)
print('SECTION 6 — FLAGS')
print(SEP)

flag_descriptions = {
    'A': 'Softmax_Part_A < 5pp on monotonic dataset [expected: ~50pp from pilot]',
    'B': 'Softmax_Part_A < 0 on monotonic dataset [Y+Adam outperforms X_diff+Adam]',
    'C': 'Crossover k mismatch between softmax and MLP for same dataset',
    'D': 'SGD_vs_Adam_gap on Y > 20pp [optimizer choice dominates]',
}

if not flags:
    print('\n  No flags fired. All results consistent with expectations.')
else:
    # Group by flag type
    from collections import defaultdict
    by_type = defaultdict(list)
    for f in flags:
        by_type[f[0]].append(f)

    for ftype in sorted(by_type.keys()):
        print(f'\n  FLAG {ftype}: {flag_descriptions[ftype]}')
        print(f'  {"-"*70}')
        for _, ds, k, msg, nums in by_type[ftype]:
            k_str = f'k={k}' if k is not None else 'all k'
            print(f'  {ds}, {k_str}: {msg}')
            print(f'    Numbers: {nums}')
        print(f'\n  What it means and what follow-up would be needed:')
        if ftype == 'A':
            print('  A small Softmax_Part_A on a monotonic dataset means X_diff and Y')
            print('  perform similarly for softmax regression — the feature-label')
            print('  misalignment is weaker than expected. Possible causes: the dataset')
            print('  at this k has low feature dimensionality relative to training size,')
            print('  or the class signal is unusually well-concentrated in U.')
            print('  Follow-up: check d_eff and train set size ratio.')
        elif ftype == 'B':
            print('  Y+Adam outperforming X_diff+Adam would contradict the pilot finding')
            print('  and the span-identity argument. Possible causes: numerical instability')
            print('  in X_diff at high k (near-rank-deficient), or the standard Euclidean')
            print('  metric on Y happens to align with class boundaries at this k.')
            print('  Follow-up: verify d_eff, check if X_diff is near-rank-deficient.')
        elif ftype == 'C':
            print('  A crossover k mismatch between softmax and MLP is the most')
            print('  scientifically interesting finding. It means the nonlinearity (or')
            print('  depth) of the MLP changes WHEN Y becomes competitive with X_diff,')
            print('  not just HOW MUCH it helps. Follow-up: examine training curves at')
            print('  the disagreement k to understand what the MLP does differently.')
        elif ftype == 'D':
            print('  A large Adam advantage on Y (>20pp) would suggest the optimizer')
            print('  geometry matters more than the 2-4pp pilot difference implied.')
            print('  Follow-up: check whether this is dataset-specific or generalizes.')

# ────────────────────────────────────────────────────────────────────────────
# Section 7 — Connection to main paper findings
# ────────────────────────────────────────────────────────────────────────────
print('\n' + SEP)
print('SECTION 7 — CONNECTION TO MAIN PAPER FINDINGS')
print(SEP)

print("""
  The paper identifies three behavioral regimes based on Part A and Part B:

  REGIME 1 — MONOTONIC (cora, citeseer, pubmed, coauthor-cs, coauthor-physics):
    MLP Part A > 0 at all k: SGC+MLP consistently outperforms Restricted+Std.
    Softmax replication check: does Softmax_Part_A > 0 at all k on these datasets?
""")
for ds in sorted(MONOTONIC):
    drows_ds = ds_rows(ds)
    all_positive = all(r['softmax_part_a'] > 0 for r in drows_ds if 'softmax_part_a' in r)
    min_sa = min((r['softmax_part_a'] for r in drows_ds), default=None)
    print(f'    {ds:<22}: all Softmax_Part_A > 0? {all_positive}  '
          f'(min = {fmt(min_sa)} pp)')

print("""
  If replication holds: the monotonic regime is NOT MLP-specific — it exists at
  the linear level. The difficulty gradient descent has with U is the same for
  softmax regression as for 3-layer MLP.
  If replication fails at any k: the MLP's nonlinearity helps U more than X_diff
  at that k, partially closing the gap that softmax cannot close.

  REGIME 2 — CROSSOVER (amazon-computers, amazon-photo, ogbn-arxiv, wikics):
    MLP Part A changes sign at some k: U eventually HELPS relative to X_diff.
    Softmax replication check: does Softmax_Part_A also change sign, at same k?
""")
for ds in sorted(CROSSOVER):
    drows_ds  = ds_rows(ds)
    soft_co   = find_crossover(drows_ds, 'softmax_part_a')
    mlp_co    = find_crossover(drows_ds, 'mlp_part_a')
    match     = 'MATCH' if soft_co == mlp_co else f'MISMATCH (soft={soft_co}, mlp={mlp_co})'
    print(f'    {ds:<22}: soft_co=k={soft_co}  mlp_co=k={mlp_co}  → {match}')

print("""
  If crossovers match: the crossover is determined by the feature geometry alone.
  The MLP nonlinearity does not change WHEN U becomes beneficial.
  If crossovers mismatch (Flag C): the nonlinearity changes the timing of benefit,
  implying the MLP is doing something with U's high-k structure that softmax cannot.

  REGIME 3 — DEFERRED CROSSOVER (none explicitly listed, but some crossover datasets
  may show very late crossovers only in MLP that softmax never reaches):
  A softmax that never crosses over despite MLP crossing at high k would imply the
  crossover requires the MLP's ability to compose nonlinear functions of eigenvectors —
  confirming that the benefit of U at high k is fundamentally nonlinear.
""")

print('\n' + SEP)
print('END OF EXP 9 ANALYSIS REPORT')
print(SEP)
