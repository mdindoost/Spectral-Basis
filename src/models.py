"""
Model architectures and training functions for spectral basis experiments.

Models:
  SGC                  — logistic regression baseline
  StandardMLP          — plain 3-layer MLP (no normalization, no dropout)
  RowNormMLP           — row-normalized MLP without bias
  CosineRowNormMLP     — row-normalized MLP with cosine classifier
  LogMagnitudeMLP      — RowNorm augmented with log-magnitude channel
  DualStreamMLP        — separate direction and magnitude branches
  SpectralRowNormMLP   — eigenvalue-weighted RowNorm (alpha sweep)
  NestedSpheresMLP     — eigenvalue weighting + magnitude augmentation (sequential MLP)
  NestedSpheresClassifier — full nested spheres with beta-scaled log-magnitude

Training helpers (legacy, used by older investigation scripts):
  train_simple         — basic training loop
  train_with_selection — training with best-val-loss / best-val-acc checkpointing
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Model Architectures
# ============================================================================

class SGC(nn.Module):
    """SGC: Logistic regression with bias"""
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()
        self.W = nn.Linear(nfeat, nclass, bias=True)
        torch.nn.init.xavier_normal_(self.W.weight)

    def forward(self, x):
        return self.W(x)


class StandardMLP(nn.Module):
    """Standard MLP with bias and no normalization.

    WARNING: No dropout — causes severe overfitting on small datasets (e.g. Cora:
    train_acc ~100%, test_acc ~38%). This creates an architectural confound when
    comparing against RowNormMLP (which benefits implicitly from row-normalization
    acting as a form of regularisation). Do NOT report StandardMLP test accuracy as
    a general baseline; use it only to measure the specific RowNorm effect (Part B).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RowNormMLP(nn.Module):
    """Row-normalized MLP without bias (radial/angular)"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = F.relu(self.fc1(x))
        x = F.normalize(x, p=2, dim=1)
        x = F.relu(self.fc2(x))
        x = F.normalize(x, p=2, dim=1)
        x = self.fc3(x)
        return x


class CosineRowNormMLP(nn.Module):
    """Row-normalized MLP with cosine classifier"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = F.relu(self.fc1(x))
        x = F.normalize(x, p=2, dim=1)
        x = F.relu(self.fc2(x))
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.fc3.weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        logits = self.scale * (x @ w.t())
        return logits


class LogMagnitudeMLP(nn.Module):
    """
    Log-Magnitude Augmented MLP
    Augments RowNorm features with log-magnitude
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(LogMagnitudeMLP, self).__init__()

        # MLP with one extra input (log-magnitude)
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Row normalization
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / (x_norm + 1e-8)

        # Log-magnitude
        log_magnitude = torch.log(x_norm + 1e-8)

        # Augment
        x_augmented = torch.cat([x_normalized, log_magnitude], dim=1)

        # MLP
        x = F.relu(self.fc1(x_augmented))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class DualStreamMLP(nn.Module):
    """Dual-Stream MLP with separate direction and magnitude branches.

    Direction stream: input_dim -> hidden_dir -> hidden_dir//2 (with Dropout 0.5)
    Magnitude stream: 1 -> hidden_mag -> hidden_mag (no dropout)
    Classifier:       (hidden_dir//2 + hidden_mag) -> num_classes
    """
    def __init__(self, input_dim, hidden_dir, hidden_mag, num_classes):
        super().__init__()

        self.mlp_direction = nn.Sequential(
            nn.Linear(input_dim, hidden_dir),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dir, hidden_dir // 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.mlp_magnitude = nn.Sequential(
            nn.Linear(1, hidden_mag),
            nn.ReLU(),
            nn.Linear(hidden_mag, hidden_mag)
        )

        self.classifier = nn.Linear(hidden_dir // 2 + hidden_mag, num_classes)

    def forward(self, X):
        M = torch.norm(X, dim=1, keepdim=True)
        X_norm = X / (M + 1e-10)
        log_M  = torch.log(M + 1e-10)
        h_dir = self.mlp_direction(X_norm)
        h_mag = self.mlp_magnitude(log_M)
        return self.classifier(torch.cat([h_dir, h_mag], dim=1))


class SpectralRowNormMLP(nn.Module):
    """
    Eigenvalue-weighted row-normalisation MLP (spectral alpha sweep, Section 4).

    Applies V_weighted = V * |eigenvalues|^alpha, then row-normalises the result
    before feeding into a 3-layer MLP with bias and Dropout(0.5).

    NOTE: This architecture differs from RowNormMLP in three ways:
      - Has bias=True in all Linear layers (RowNormMLP uses bias=False)
      - Applies normalisation at the INPUT only (RowNormMLP re-normalises after each layer)
      - Has Dropout(0.5) (RowNormMLP has no dropout)
    Therefore alpha=0 does NOT reproduce RowNormMLP; see C-2 in CODE_REVIEW.md.

    Args:
        input_dim: Number of input features (= d_effective eigenvectors)
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        eigenvalues: Eigenvalues (shape: d_effective) for weighting
        alpha: Weighting exponent. alpha=0 gives uniform weights (all ones).
    """
    def __init__(self, input_dim, hidden_dim, num_classes, eigenvalues, alpha=0.5):
        super(SpectralRowNormMLP, self).__init__()

        self.alpha = alpha

        # Compute eigenvalue weights
        # Handle alpha=0 separately to avoid numerical issues
        if abs(alpha) < 1e-8:
            self.eigenvalue_weights = torch.ones(input_dim)
        else:
            # Clamp to a reasonable floor before raising to alpha (especially negative alpha).
            # Without clamping, near-zero eigenvalues give weights ~1/λ → very large numbers
            # for alpha=-1, making the model collapse onto a single eigenvector direction.
            eigenvalues_safe = torch.clamp(torch.abs(eigenvalues), min=1e-4)
            self.eigenvalue_weights = eigenvalues_safe ** alpha

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('weights', self.eigenvalue_weights)

        # MLP architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Step 1: Weight by eigenvalues (nested spheres)
        x_weighted = x * self.weights

        # Step 2: Row normalization
        x_norm = torch.norm(x_weighted, p=2, dim=1, keepdim=True)
        x_normalized = x_weighted / (x_norm + 1e-8)

        # Step 3: MLP
        x = F.relu(self.fc1(x_normalized))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class NestedSpheresMLP(nn.Module):
    """
    Full Nested Spheres architecture combining:
    1. Eigenvalue weighting (spectral structure)
    2. Magnitude preservation (node importance)

    V_weighted = V * (eigenvalues ** alpha)
    M = ||V_weighted||_row
    V_normalized = V_weighted / M
    X_augmented = [V_normalized, beta * log(M)]
    """
    def __init__(self, input_dim, hidden_dim, num_classes, eigenvalues, alpha=0.5, beta=1.0):
        super().__init__()

        # Eigenvalue weighting — clamp to 1e-4 (same floor as SpectralRowNormMLP)
        # to prevent near-zero eigenvalues from exploding when alpha < 0.
        if abs(alpha) < 1e-8:
            eigenvalue_weights = torch.ones(input_dim)
        else:
            eigenvalues_safe   = torch.clamp(torch.abs(eigenvalues), min=1e-4)
            eigenvalue_weights = eigenvalues_safe ** alpha

        self.register_buffer('eigenvalue_weights', eigenvalue_weights)
        self.beta = beta

        # MLP (input_dim + 1 because we add log-magnitude)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, X):
        # Weight by eigenvalues (nested spheres)
        X_weighted = X * self.eigenvalue_weights

        # Compute magnitude
        M = torch.norm(X_weighted, dim=1, keepdim=True)

        # Normalize
        X_normalized = X_weighted / (M + 1e-10)

        # Augment with log-magnitude
        log_M = torch.log(M + 1e-10)
        X_augmented = torch.cat([X_normalized, self.beta * log_M], dim=1)

        return self.mlp(X_augmented)


class NestedSpheresClassifier(nn.Module):
    """
    Complete nested spheres architecture combining:
    1. Eigenvalue weighting (spectral structure)
    2. Magnitude preservation (node importance)

    Architecture:
        V_weighted = V * (eigenvalues ** alpha)
        M = log(||V_weighted||_row)
        V_normalized = V_weighted / ||V_weighted||_row
        X_augmented = [V_normalized, beta * M]
        output = MLP(X_augmented)

    Args:
        input_dim: Number of features
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        eigenvalues: Eigenvalues corresponding to eigenvectors
        alpha: Eigenvalue weighting exponent
        beta: Magnitude feature scaling factor
    """
    def __init__(self, input_dim, hidden_dim, num_classes, eigenvalues,
                 alpha=0.5, beta=1.0):
        super(NestedSpheresClassifier, self).__init__()

        self.alpha = alpha
        self.beta = beta

        # Compute eigenvalue weights — clamp to 1e-4 (same floor as SpectralRowNormMLP)
        # to prevent near-zero eigenvalues from exploding when alpha < 0.
        if abs(alpha) < 1e-8:
            self.eigenvalue_weights = torch.ones(input_dim)
        else:
            eigenvalues_safe        = torch.clamp(torch.abs(eigenvalues), min=1e-4)
            self.eigenvalue_weights = eigenvalues_safe ** alpha

        self.register_buffer('weights', self.eigenvalue_weights)

        # MLP with one extra input (log-magnitude)
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Step 1: Weight by eigenvalues (nested spheres)
        x_weighted = x * self.weights

        # Step 2: Compute magnitude
        magnitude = torch.norm(x_weighted, p=2, dim=1, keepdim=True)

        # Step 3: Row normalization
        x_normalized = x_weighted / (magnitude + 1e-8)

        # Step 4: Log-magnitude
        log_magnitude = torch.log(magnitude + 1e-8)

        # Step 5: Augment with scaled log-magnitude
        x_augmented = torch.cat([x_normalized, self.beta * log_magnitude], dim=1)

        # Step 6: MLP
        x = F.relu(self.fc1(x_augmented))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# ============================================================================
# Training Functions (legacy — used by older investigation scripts)
# ============================================================================

def train_simple(model, train_loader, X_val, y_val, X_test, y_test,
                 epochs=200, lr=0.01, weight_decay=5e-4, device='cpu', verbose=True):
    """
    Simple training loop for basic experiments
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).argmax(1)
            val_acc = (val_pred == y_val).float().mean().item()
            val_accs.append(val_acc)
            train_losses.append(total_loss / len(train_loader))

        if verbose and (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Loss: {train_losses[-1]:.4f} | Val Acc: {val_acc:.4f}')

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(1)
        test_acc = (test_pred == y_test).float().mean().item()

    return {
        'model': model,
        'train_losses': train_losses,
        'val_accs': val_accs,
        'test_acc': test_acc
    }


def train_with_selection(model, train_loader, X_val, y_val, X_test, y_test,
                         epochs=200, lr=0.01, weight_decay=5e-4, device='cpu', verbose=True):
    """
    Training with model selection (best val loss and best val acc)
    Used in Investigation 2
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_accs = [], [], []

    best_loss = float('inf')
    best_acc = -1.0
    best_state_by_loss = None
    best_state_by_acc = None
    best_ep_loss = -1
    best_ep_acc = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val).item()
            val_pred = val_output.argmax(1)
            val_acc = (val_pred == y_val).float().mean().item()

        train_losses.append(total_loss / len(train_loader))
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state_by_loss = copy.deepcopy(model.state_dict())
            best_ep_loss = epoch

        if val_acc > best_acc:
            best_acc = val_acc
            best_state_by_acc = copy.deepcopy(model.state_dict())
            best_ep_acc = epoch

        if verbose and (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    # Test at best val loss
    model.load_state_dict(best_state_by_loss)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(1)
        test_acc_best_loss = (test_pred == y_test).float().mean().item()

    # Test at best val acc
    model.load_state_dict(best_state_by_acc)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(1)
        test_acc_best_acc = (test_pred == y_test).float().mean().item()

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_loss': best_loss,
        'best_val_acc': best_acc,
        'best_epoch_loss': best_ep_loss,
        'best_epoch_acc': best_ep_acc,
        'test_at_best_val_loss': test_acc_best_loss,
        'test_at_best_val_acc': test_acc_best_acc
    }
