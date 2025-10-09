"""
Shared utilities for spectral basis experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader, TensorDataset
import copy


class StandardMLP(nn.Module):
    """Standard 3-layer MLP with bias and ReLU activations"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=True)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RowNormMLP(nn.Module):
    """Row-normalized MLP without bias (radial projection)"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)
    
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)  # L2 normalize rows
        x = F.relu(self.fc1(x))
        x = F.normalize(x, p=2, dim=1)
        x = F.relu(self.fc2(x))
        x = F.normalize(x, p=2, dim=1)
        x = self.fc3(x)
        return x


class CosineRowNormMLP(nn.Module):
    """Row-normalized MLP with cosine classifier and learnable temperature"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))  # learnable temperature
    
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = F.relu(self.fc1(x))
        x = F.normalize(x, p=2, dim=1)
        x = F.relu(self.fc2(x))
        x = F.normalize(x, p=2, dim=1)
        
        # Cosine classifier (no bias)
        w = F.normalize(self.fc3.weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        logits = self.scale * (x @ w.t())
        return logits


def build_graph_matrices(edge_index, num_nodes):
    """
    Build adjacency and Laplacian matrices from edge index
    
    Returns:
        adj: Sparse adjacency matrix (with self-loops)
        D: Sparse degree matrix
        L: Sparse Laplacian matrix (D - adj)
    """
    edges = edge_index.T
    
    # Build sparse adjacency
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0], dtype=np.float64), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes)
    )
    
    # Symmetrize and add self-loops
    adj = adj.maximum(adj.T)
    adj = adj + sp.eye(num_nodes, dtype=np.float64)
    
    # Degree and Laplacian
    deg = np.array(adj.sum(axis=1)).ravel()
    D = sp.diags(deg)
    L = D - adj
    
    return adj, D, L


def train_simple(model, train_loader, X_val, y_val, X_test, y_test,
                 epochs=60, lr=1e-2, weight_decay=5e-4, device='cpu', verbose=True):
    """
    Simple training loop (for Investigation 1)
    Returns final test accuracy only
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accs = []
    
    for ep in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            opt.zero_grad()
            out = model(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_pred = val_out.argmax(1)
            val_acc = (val_pred == y_val).float().mean().item()
        
        train_losses.append(total_loss / max(1, len(train_loader)))
        val_accs.append(val_acc)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f'Epoch {ep+1}/{epochs}  TrainLoss={train_losses[-1]:.4f}  ValAcc={val_acc:.4f}')
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(1)
        test_acc = (test_pred == y_test).float().mean().item()
    
    return {
        'train_losses': train_losses,
        'val_accs': val_accs,
        'test_acc': test_acc
    }


def train_with_selection(model, train_loader, X_val, y_val, X_test, y_test,
                         epochs=200, lr=1e-2, weight_decay=5e-4, device='cpu', verbose=True):
    """
    Training with model selection by best val loss AND best val acc (for Investigation 2)
    Returns test accuracy at both checkpoints
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    val_accs = []
    
    best_loss = float('inf')
    best_acc = -1.0
    best_state_by_loss = None
    best_state_by_acc = None
    best_ep_loss = -1
    best_ep_acc = -1
    
    for ep in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            opt.zero_grad()
            out = model(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = crit(val_out, y_val).item()
            val_pred = val_out.argmax(1)
            val_acc = (val_pred == y_val).float().mean().item()
        
        train_losses.append(total_loss / max(1, len(train_loader)))
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Track best models
        if val_loss < best_loss:
            best_loss = val_loss
            best_state_by_loss = copy.deepcopy(model.state_dict())
            best_ep_loss = ep
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state_by_acc = copy.deepcopy(model.state_dict())
            best_ep_acc = ep
        
        if verbose and (ep + 1) % 20 == 0:
            print(f'Epoch {ep+1}/{epochs}  TrainLoss={train_losses[-1]:.4f}  '
                  f'ValLoss={val_loss:.4f}  ValAcc={val_acc:.4f}')
    
    # Evaluate test at best-by-loss checkpoint
    model.load_state_dict(best_state_by_loss)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).argmax(1)
        test_acc_best_loss = (test_pred == y_test).float().mean().item()
    
    # Evaluate test at best-by-acc checkpoint
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
        'test_at_best_val_acc': test_acc_best_acc,
    }
