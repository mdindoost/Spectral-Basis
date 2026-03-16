"""
Exp 9 Whitening Check — metric mismatch hypothesis test.

Hypothesis: U performs poorly under gradient descent because it is orthonormal
in the D-weighted inner product (U^T D U = I) but NOT in the standard Euclidean
inner product (U^T U ≠ I). The Gram matrix U^T U is ill-conditioned, which
misaligns the gradient descent geometry.

Test: compute U_whitened = U @ (U^T U)^{-1/2}, which enforces
      U_whitened^T U_whitened = I  (Euclidean orthonormality).
      Then run Adam lr=0.01, 500 epochs, 15 seeds on Cora k=1.

Prediction:
  - If metric mismatch is the sole cause: U_whitened test_acc → X_diff+Adam (70.35%)
  - If something else is at play: U_whitened test_acc stays well below 70.35%

Dataset: Cora k=1 (fixed LCC split).
Whitening: computed on all n rows of U (U is a global graph-spectral object;
           the whitening matrix W = (U^T U)^{-1/2} is a d_eff × d_eff linear map
           and does not use label information, so no leakage concern applies).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Load precomputed arrays ──────────────────────────────────────────────────
d = os.path.join(_REPO_ROOT, 'data', 'cora_fixed_lcc')
U       = np.load(os.path.join(d, 'Y_k1.npy'))        # (2485, 1427)
labels  = np.load(os.path.join(d, 'labels.npy'))
split   = np.load(os.path.join(d, 'split_idx.npy'), allow_pickle=True).item()
tr, va, te = split['train'], split['val'], split['test']

num_classes = len(np.unique(labels))
n, d_eff    = U.shape
print(f'U shape: {U.shape}  |  classes: {num_classes}')

# ── Compute U_whitened = U @ (U^T U)^{-1/2} ─────────────────────────────────
# Use eigendecomposition of the symmetric PSD matrix G = U^T U.
# G = V Λ V^T  →  G^{-1/2} = V Λ^{-1/2} V^T
G            = U.T @ U                          # (d_eff, d_eff)
eigvals, V   = np.linalg.eigh(G)               # ascending eigenvalues

# Report conditioning before inverting — critical for understanding U^T U
print(f'\nU^T U eigenvalue range: min={eigvals.min():.4e}  max={eigvals.max():.4e}')
print(f'Condition number (U^T U): {eigvals.max() / eigvals.min():.4e}')

# Floor small eigenvalues for numerical safety (same 1e-10 floor as graph_utils.py)
eps          = 1e-10 * eigvals.max()
eigvals_safe = np.maximum(eigvals, eps)
G_inv_sqrt   = V @ np.diag(1.0 / np.sqrt(eigvals_safe)) @ V.T

U_whitened   = U @ G_inv_sqrt                  # (n, d_eff)

# ── Verify U_whitened^T U_whitened ≈ I ───────────────────────────────────────
GW        = U_whitened.T @ U_whitened
gram_err  = np.max(np.abs(GW - np.eye(d_eff)))
print(f'U_whitened^T U_whitened = I  error (max |G - I|): {gram_err:.2e}')
if gram_err > 1e-4:
    print('WARNING: Gram error is large — whitening may be numerically unstable.')
else:
    print('Gram check PASSED.')

# ── Model and training (identical to exp9_softmax_convergence.py) ─────────────
class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=True)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    def forward(self, x):
        return self.fc(x)

def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

NUM_SEEDS = 15
EPOCHS    = 500
LR        = 0.01
WD        = 0.0

Xtr_t = torch.FloatTensor(U_whitened[tr]).to(device)
Xva_t = torch.FloatTensor(U_whitened[va]).to(device)
Xte_t = torch.FloatTensor(U_whitened[te]).to(device)
ytr_t = torch.LongTensor(labels[tr]).to(device)
yva_t = torch.LongTensor(labels[va]).to(device)
yte_t = torch.LongTensor(labels[te]).to(device)

final_tes = []
peak_vas  = []

for seed in range(NUM_SEEDS):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = SoftmaxRegression(d_eff, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss()
    best_va   = 0.0

    for _ep in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xtr_t), ytr_t)
        loss.backward()
        optimizer.step()
        va_acc = accuracy(model, Xva_t, yva_t)
        if va_acc > best_va:
            best_va = va_acc

    final_tes.append(accuracy(model, Xte_t, yte_t))
    peak_vas.append(best_va)

# ── Report ────────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('RESULT — U_whitened + Adam, Cora k=1, 500 epochs')
print('=' * 60)
print(f'  Test acc (final ep 500):  {np.mean(final_tes)*100:.2f}% ± {np.std(final_tes)*100:.2f}%')
print(f'  Peak val acc:             {np.mean(peak_vas)*100:.2f}% ± {np.std(peak_vas)*100:.2f}%')
print(f'  Per-seed test accs: {[round(v*100,2) for v in final_tes]}')
print()
print('  Reference baselines (from exp9_softmax_convergence.py):')
print('    X_diff + Adam, ep 500:  70.35% ± 0.34%')
print('    Y (raw U) + Adam, ep 500: 17.59% ± 0.12%')
print()
gap_to_xdiff = 70.35 - np.mean(final_tes) * 100
gap_to_rawU  = np.mean(final_tes) * 100 - 17.59
print(f'  Gap vs X_diff+Adam:  {gap_to_xdiff:+.2f} pp')
print(f'  Gain vs raw U+Adam:  {gap_to_rawU:+.2f} pp')
print()
if gap_to_xdiff < 2.0:
    print('  VERDICT: Metric mismatch hypothesis CONFIRMED.')
    print('  U_whitened converges to X_diff accuracy → Euclidean orthonormality')
    print('  is sufficient to recover the representational capacity of X_diff.')
elif gap_to_rawU > 30.0:
    print('  VERDICT: Metric mismatch is A MAJOR factor but may not be the only one.')
    print(f'  Whitening recovers {gap_to_rawU:.1f} pp of the {70.35-17.59:.1f} pp gap.')
    print('  Remaining gap suggests additional structure (e.g. feature scale,')
    print('  eigenvalue distribution) also contributes.')
else:
    print('  VERDICT: Metric mismatch is NOT the primary explanation.')
    print('  Whitening gives little gain over raw U — something else is at play.')
print('=' * 60)
