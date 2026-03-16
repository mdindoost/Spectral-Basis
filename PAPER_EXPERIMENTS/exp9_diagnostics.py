"""
Exp 9 Diagnostics — three targeted checks on Cora k=1 and Amazon-Computers k=1.

Check 1: U + Adam, Cora k=1, extended to 2000 epochs.
         Report test_acc at epochs 500, 1000, 1500, 2000.
         Goal: does U converge toward X_diff accuracy or plateau?

Check 2: U + SGD, Cora k=1, lr ∈ {0.001, 0.01, 0.1}.
         Report final test_acc and Ep→90% for each lr.
         Goal: does a different lr rescue SGD on U?

Check 3: U + SGD, Amazon-Computers k=1.
         Report ||∇L|| (total gradient norm over all parameters) at
         epochs 1, 2, 5, 10. Measured after loss.backward(), before step().
         Goal: confirm whether gradients are near-zero or nonzero but misdirected.

All checks use:
  - 15 seeds (same as exp9)
  - Fixed split and LCC-remapped indices from precomputed data/ arrays
  - SoftmaxRegression model (single Linear layer, bias=True, Xavier init)
  - Full-batch, CrossEntropyLoss, weight_decay=0
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, _SRC_DIR)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')


# ============================================================================
# Helpers
# ============================================================================

class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=True)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)


def load_data(dataset, k):
    """Load precomputed arrays saved by exp9_softmax_convergence.py Step 1."""
    d = os.path.join(_REPO_ROOT, 'data', f'{dataset}_fixed_lcc')
    Y        = np.load(os.path.join(d, f'Y_k{k}.npy'))
    X_diff   = np.load(os.path.join(d, f'X_diff_k{k}.npy'))
    labels   = np.load(os.path.join(d, 'labels.npy'))
    split    = np.load(os.path.join(d, 'split_idx.npy'), allow_pickle=True).item()
    return X_diff, Y, labels, split['train'], split['val'], split['test']


def to_tensors(X, labels, tr, va, te, dev):
    """Slice arrays by split and move to device."""
    Xtr = torch.FloatTensor(X[tr]).to(dev)
    Xva = torch.FloatTensor(X[va]).to(dev)
    Xte = torch.FloatTensor(X[te]).to(dev)
    ytr = torch.LongTensor(labels[tr]).to(dev)
    yva = torch.LongTensor(labels[va]).to(dev)
    yte = torch.LongTensor(labels[te]).to(dev)
    return Xtr, Xva, Xte, ytr, yva, yte


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def grad_norm(model):
    """Total L2 norm of all gradients (after loss.backward())."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


NUM_SEEDS = 15


# ============================================================================
# Check 1: U + Adam, Cora k=1, 2000 epochs
# Report test_acc at checkpoints {500, 1000, 1500, 2000}
# ============================================================================

print('=' * 65)
print('CHECK 1: U + Adam, Cora k=1, extended to 2000 epochs')
print('=' * 65)

X_diff_c, Y_c, labels_c, tr_c, va_c, te_c = load_data('cora', 1)
num_classes_c = len(np.unique(labels_c))
d_eff_c = Y_c.shape[1]

CHECKPOINTS = [500, 1000, 1500, 2000]

# Accumulate test_acc at each checkpoint across seeds
ckpt_accs = {ep: [] for ep in CHECKPOINTS}

for seed in range(NUM_SEEDS):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, Xva, Xte, ytr, yva, yte = to_tensors(Y_c, labels_c, tr_c, va_c, te_c, device)

    model = SoftmaxRegression(d_eff_c, num_classes_c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 2001):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xtr), ytr)
        loss.backward()
        optimizer.step()

        if epoch in CHECKPOINTS:
            ckpt_accs[epoch].append(accuracy(model, Xte, yte))

print(f'\n  {"Epoch":>6}  {"Mean Test%":>10}  {"±Std":>7}  {"Min":>7}  {"Max":>7}')
print(f'  {"-"*45}')
for ep in CHECKPOINTS:
    accs = ckpt_accs[ep]
    print(f'  {ep:>6}  {np.mean(accs)*100:>9.2f}%  '
          f'±{np.std(accs)*100:>5.2f}%  '
          f'{min(accs)*100:>6.2f}%  '
          f'{max(accs)*100:>6.2f}%')

# Context: X_diff + Adam at epoch 500 was 70.35% (from exp9 run)
print(f'\n  Reference — X_diff + Adam at epoch 500: 70.35%')
print(f'  Reference — Y + Adam at epoch 500 (from exp9): 17.59%')
final_mean = np.mean(ckpt_accs[2000]) * 100
gap_to_xdiff = 70.35 - final_mean
print(f'  Gap remaining at epoch 2000: {gap_to_xdiff:.2f} pp below X_diff')


# ============================================================================
# Check 2: U + SGD, Cora k=1, lr ∈ {0.001, 0.01, 0.1}
# Report final test_acc (epoch 500) and Ep→90% for each lr
# ============================================================================

print('\n' + '=' * 65)
print('CHECK 2: U + SGD, Cora k=1, lr sweep {0.001, 0.01, 0.1}')
print('=' * 65)

LR_VALUES = [0.001, 0.01, 0.1]
EPOCHS_C2  = 500

def ep_to_pct_of_peak(val_accs, pct):
    target = pct * max(val_accs)
    for i, v in enumerate(val_accs):
        if v >= target:
            return i + 1
    return None

print(f'\n  {"lr":>8}  {"Test%":>9}  {"±Std":>6}  {"PeakVal%":>9}  {"Ep→90%":>8}  {"Ep→95%":>8}')
print(f'  {"-"*65}')

for lr in LR_VALUES:
    final_tes = []
    peak_vas  = []
    ep90s, ep95s = [], []

    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)

        Xtr, Xva, Xte, ytr, yva, yte = to_tensors(Y_c, labels_c, tr_c, va_c, te_c, device)

        model = SoftmaxRegression(d_eff_c, num_classes_c).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()

        val_accs = []
        for _epoch in range(EPOCHS_C2):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(Xtr), ytr)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                va_acc = (model(Xva).argmax(1) == yva).float().mean().item()
            val_accs.append(va_acc)

        final_tes.append(accuracy(model, Xte, yte))
        peak_vas.append(max(val_accs))

        e90 = ep_to_pct_of_peak(val_accs, 0.90)
        e95 = ep_to_pct_of_peak(val_accs, 0.95)
        if e90 is not None: ep90s.append(e90)
        if e95 is not None: ep95s.append(e95)

    ep90_str = f'{np.mean(ep90s):.1f}' if ep90s else 'N/A'
    ep95_str = f'{np.mean(ep95s):.1f}' if ep95s else 'N/A'
    print(f'  {lr:>8.3f}  {np.mean(final_tes)*100:>8.2f}%  '
          f'±{np.std(final_tes)*100:>4.2f}%  '
          f'{np.mean(peak_vas)*100:>8.2f}%  '
          f'{ep90_str:>8}  {ep95_str:>8}')

print(f'\n  Reference — X_diff + SGD at lr=0.01, epoch 500: 61.65%')


# ============================================================================
# Check 3: U + SGD, Amazon-Computers k=1
# Gradient norm ||∇L|| at epochs 1, 2, 5, 10
# Report mean ± std over 15 seeds
# ============================================================================

print('\n' + '=' * 65)
print('CHECK 3: U + SGD, Amazon-Computers k=1 — gradient norms')
print('=' * 65)

X_diff_a, Y_a, labels_a, tr_a, va_a, te_a = load_data('amazon-computers', 1)
num_classes_a = len(np.unique(labels_a))
d_eff_a = Y_a.shape[1]

GRAD_EPOCHS = {1, 2, 5, 10}

# Also track test_acc at those epochs to confirm the "stuck at 37.71%" observation
grad_norms_by_ep  = {ep: [] for ep in sorted(GRAD_EPOCHS)}
test_accs_by_ep   = {ep: [] for ep in sorted(GRAD_EPOCHS)}

for seed in range(NUM_SEEDS):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, Xva, Xte, ytr, yva, yte = to_tensors(Y_a, labels_a, tr_a, va_a, te_a, device)

    model = SoftmaxRegression(d_eff_a, num_classes_a).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, max(GRAD_EPOCHS) + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xtr), ytr)
        loss.backward()

        if epoch in GRAD_EPOCHS:
            gn = grad_norm(model)
            te = accuracy(model, Xte, yte)
            grad_norms_by_ep[epoch].append(gn)
            test_accs_by_ep[epoch].append(te)

        optimizer.step()

print(f'\n  {"Epoch":>6}  {"||∇L|| mean":>12}  {"±Std":>10}  '
      f'{"||∇L|| min":>12}  {"||∇L|| max":>12}  {"Test%":>7}')
print(f'  {"-"*70}')
for ep in sorted(GRAD_EPOCHS):
    gns = grad_norms_by_ep[ep]
    tes = test_accs_by_ep[ep]
    print(f'  {ep:>6}  {np.mean(gns):>12.6f}  '
          f'±{np.std(gns):>8.6f}  '
          f'{min(gns):>12.6f}  '
          f'{max(gns):>12.6f}  '
          f'{np.mean(tes)*100:>6.2f}%')

# Also report weight norm to give context: is gradient small relative to weights?
# Re-run one seed to capture weight norms alongside gradient norms
torch.manual_seed(0); np.random.seed(0)
Xtr, Xva, Xte, ytr, yva, yte = to_tensors(Y_a, labels_a, tr_a, va_a, te_a, device)
model = SoftmaxRegression(d_eff_a, num_classes_a).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0)
criterion = nn.CrossEntropyLoss()

print(f'\n  Weight / loss context at seed=0:')
print(f'  {"Epoch":>6}  {"Loss":>10}  {"||W||":>10}  {"||∇L||":>10}  {"||∇L||/||W||":>14}')
print(f'  {"-"*56}')
for epoch in range(1, max(GRAD_EPOCHS) + 1):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(Xtr), ytr)
    loss.backward()

    if epoch in GRAD_EPOCHS:
        gn = grad_norm(model)
        wn = sum(p.data.norm(2).item()**2 for p in model.parameters())**0.5
        print(f'  {epoch:>6}  {loss.item():>10.6f}  {wn:>10.6f}  {gn:>10.6f}  {gn/wn:>14.6f}')

    optimizer.step()

print(f'\n  Reference: X_diff + SGD at Amazon-Computers k=1 → 74.04% at epoch 500')
print(f'  Y + SGD was stuck at 37.71% (std=0.00) across all seeds.')

print('\n' + '=' * 65)
print('ALL CHECKS COMPLETE')
print('=' * 65)
