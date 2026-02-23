"""
Utility functions for spectral basis experiments
Extended to support larger datasets: WikiCS, Amazon, Coauthor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import networkx as nx

# Dataset loaders
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.datasets import Planetoid, WikiCS, Amazon, Coauthor
from torch_geometric.utils import to_undirected

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
        
        # Eigenvalue weighting
        if abs(alpha) < 1e-8:
            eigenvalue_weights = torch.ones(input_dim)
        else:
            eigenvalues_safe = torch.abs(eigenvalues) + 1e-8
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
        
        # Compute eigenvalue weights
        if abs(alpha) < 1e-8:
            self.eigenvalue_weights = torch.ones(input_dim)
        else:
            eigenvalues_safe = torch.abs(eigenvalues) + 1e-8
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
# Dataset Loading
# ============================================================================

def load_dataset(dataset_name, root='./dataset'):
    """
    Load dataset and return graph structure, features, labels, and splits
    
    Supported datasets:
    - OGB: ogbn-arxiv
    - Planetoid: cora, citeseer, pubmed
    - WikiCS: wikics
    - Amazon: amazon-photo, amazon-computers
    - Coauthor: coauthor-cs, coauthor-physics
    
    Returns:
        edge_index: (2, num_edges) edge list
        node_features: (num_nodes, num_features) or None
        labels: (num_nodes,) node labels
        num_nodes: int
        num_classes: int
        train_idx: training node indices
        val_idx: validation node indices
        test_idx: test node indices
    """
    dataset_name = dataset_name.lower()
    
    # ========================================================================
    # OGB Datasets
    # ========================================================================
    if dataset_name == 'ogbn-arxiv':
        dataset = NodePropPredDataset(name='ogbn-arxiv', root=root)
        graph, labels = dataset[0]
        
        edge_index = torch.from_numpy(graph['edge_index']).long()
        node_features = torch.from_numpy(graph['node_feat']).float()
        labels = torch.from_numpy(labels).squeeze().long()
        num_nodes = graph['num_nodes']
        num_classes = dataset.num_classes
        
        split_idx = dataset.get_idx_split()
        train_idx = np.asarray(split_idx['train']).reshape(-1)
        val_idx   = np.asarray(split_idx['valid']).reshape(-1)
        test_idx  = np.asarray(split_idx['test']).reshape(-1)
    
    # ========================================================================
    # Planetoid Datasets (Cora, CiteSeer, PubMed)
    # ========================================================================
    elif dataset_name in ['cora', 'citeseer', 'pubmed']:
        name_map = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed'}
        dataset = Planetoid(root=root, name=name_map[dataset_name])
        data = dataset[0]
        
        edge_index = data.edge_index
        node_features = data.x.float()
        labels = data.y.long()
        num_nodes = data.num_nodes
        num_classes = dataset.num_classes
        
        # Use standard splits
        train_idx = data.train_mask.nonzero(as_tuple=True)[0].numpy()
        val_idx = data.val_mask.nonzero(as_tuple=True)[0].numpy()
        test_idx = data.test_mask.nonzero(as_tuple=True)[0].numpy()
    
    # ========================================================================
    # WikiCS Dataset
    # ========================================================================
    elif dataset_name == 'wikics':
        dataset = WikiCS(root=f'{root}/WikiCS')
        data = dataset[0]
        
        edge_index = to_undirected(data.edge_index)
        node_features = data.x.float()
        labels = data.y.long()
        num_nodes = data.num_nodes
        num_classes = dataset.num_classes
        
        # WikiCS has 20 different train/val splits, use split 0
        train_idx = data.train_mask[:, 0].nonzero(as_tuple=True)[0].numpy()
        val_idx = data.val_mask[:, 0].nonzero(as_tuple=True)[0].numpy()
        
        # WikiCS has stopping_mask instead of test_mask
        test_idx = data.test_mask.nonzero(as_tuple=True)[0].numpy()
        
        print(f"WikiCS using split 0: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # ========================================================================
    # Amazon Datasets (Photo, Computers)
    # ========================================================================
    elif dataset_name in ['amazon-photo', 'amazon-computers']:
        name_map = {'amazon-photo': 'Photo', 'amazon-computers': 'Computers'}
        dataset = Amazon(root=f'{root}/Amazon', name=name_map[dataset_name])
        data = dataset[0]
        
        edge_index = to_undirected(data.edge_index)
        node_features = data.x.float()
        labels = data.y.long()
        num_nodes = data.num_nodes
        num_classes = dataset.num_classes
        
        # Amazon datasets don't have pre-defined splits
        # Create standard 60/20/20 split using a local RNG to avoid mutating global state.
        print(f"Creating 60/20/20 split for {dataset_name}...")
        indices = np.arange(num_nodes)
        np.random.default_rng(42).shuffle(indices)
        
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
    
    # ========================================================================
    # Coauthor Datasets (CS, Physics)
    # ========================================================================
    elif dataset_name in ['coauthor-cs', 'coauthor-physics']:
        name_map = {'coauthor-cs': 'CS', 'coauthor-physics': 'Physics'}
        dataset = Coauthor(root=f'{root}/Coauthor', name=name_map[dataset_name])
        data = dataset[0]
        
        edge_index = to_undirected(data.edge_index)
        node_features = data.x.float()
        labels = data.y.long()
        num_nodes = data.num_nodes
        num_classes = dataset.num_classes
        
        # Coauthor datasets don't have pre-defined splits
        # Create standard 60/20/20 split using a local RNG to avoid mutating global state.
        print(f"Creating 60/20/20 split for {dataset_name}...")
        indices = np.arange(num_nodes)
        np.random.default_rng(42).shuffle(indices)
        
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Convert to numpy if needed
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()
    if isinstance(node_features, torch.Tensor):
        node_features = node_features.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    return (edge_index, node_features, labels, num_nodes, num_classes,
            train_idx, val_idx, test_idx)

# ============================================================================
# Graph Construction
# ============================================================================

def build_graph_matrices(edge_index, num_nodes):
    """
    Build sparse adjacency and Laplacian matrices from edge list
    
    Returns:
        adj: scipy sparse adjacency matrix (with self-loops)
        L: scipy sparse Laplacian matrix (D - A)
        D: scipy sparse degree matrix
    """
    edges = edge_index.T
    
    # Build adjacency matrix
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0], dtype=np.float64), 
         (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes)
    )
    
    # Symmetrize and add self-loops
    adj = adj.maximum(adj.T)
    adj = adj + sp.eye(num_nodes, dtype=np.float64)
    
    # Degree and Laplacian
    deg = np.array(adj.sum(axis=1)).ravel()
    D = sp.diags(deg)
    L = D - adj
    
    return adj, L, D

def get_largest_connected_component_nx(adj):
    """Extract largest connected component"""
    G = nx.from_scipy_sparse_array(adj)
    components = list(nx.connected_components(G))
    
    if len(components) == 1:
        return np.ones(adj.shape[0], dtype=bool)
    
    largest_component = max(components, key=len)
    mask = np.zeros(adj.shape[0], dtype=bool)
    mask[list(largest_component)] = True
    
    return mask




def extract_subgraph(adj, features, labels, mask, split_idx):
    """Extract subgraph for nodes in mask"""
    node_indices = np.where(mask)[0]
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
    
    adj_sub = adj[mask][:, mask]
    features_sub = features[mask]
    labels_sub = labels[mask]
    
    split_idx_sub = None
    if split_idx is not None:
        split_idx_sub = {}
        for split_name, indices in split_idx.items():
            mask_indices = np.isin(indices, node_indices)
            old_indices = indices[mask_indices]
            new_indices = np.array([old_to_new[idx] for idx in old_indices])
            split_idx_sub[split_name] = new_indices
    
    return adj_sub, features_sub, labels_sub, split_idx_sub

def compute_sgc_normalized_adjacency(adj):
    """Compute SGC-style normalized adjacency: D^{-1/2} A~ D^{-1/2}
    where A~ = A + I. adj must already include self-loops (from build_graph_matrices).
    Do NOT add self-loops here again."""
    # adj already has self-loops added by build_graph_matrices — do not add again
    adj = sp.coo_matrix(adj)
    
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def sgc_precompute(features, adj_normalized, degree):
    """Apply SGC precomputation: A^k X"""
    for i in range(degree):
        features = adj_normalized @ features
    return features

def compute_restricted_eigenvectors(X, L, D, num_components=0):
    """
    Compute X-restricted eigenvectors via Rayleigh-Ritz procedure.
    
    Returns:
        U: Eigenvector matrix (n x d_effective)
        eigenvalues: Corresponding eigenvalues
        d_effective: Effective dimension after rank check
        ortho_error: D-orthonormality error (should be < 1e-6)
    """
    num_nodes, dimension = X.shape
    
    # Determine numerical rank via SVD (reliable for near-degenerate X at high k),
    # then use QR only for the orthonormal basis Q needed downstream.
    rank_X = np.linalg.matrix_rank(X, tol=1e-10)
    Q, _ = np.linalg.qr(X)

    if rank_X < dimension:
        Q = Q[:, :rank_X]
        dimension = rank_X
    
    d_effective = dimension
    
    # Project Laplacian and degree matrix onto column space of X
    L_r = Q.T @ (L @ Q)
    D_r = Q.T @ (D @ Q)
    
    # Symmetrize for numerical stability
    L_r = 0.5 * (L_r + L_r.T)
    D_r = 0.5 * (D_r + D_r.T)
    
    # Regularize D_r
    eps_base = 1e-10
    eps = eps_base * np.trace(D_r) / d_effective
    D_r = D_r + eps * np.eye(d_effective)
    
    # Solve generalized eigenvalue problem
    eigenvalues, V = la.eigh(L_r, D_r)
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # Skip first num_components (usually 0 for LCC)
    if num_components > 0:
        eigenvalues = eigenvalues[num_components:]
        V = V[:, num_components:]
    
    # Map back to original space
    U = Q @ V
    
    # Verify D-orthonormality: U^T D U should be identity
    G = U.T @ (D @ U)
    ortho_error = np.max(np.abs(G - np.eye(len(eigenvalues))))

    # Verify eigenvalue range: normalized Laplacian eigenvalues must lie in [0, 2]
    # Violation indicates L and D were passed swapped to this function
    eig_min = float(eigenvalues.min())
    eig_max = float(eigenvalues.max())
    if eig_min < -0.01 or eig_max > 2.1:
        import warnings
        warnings.warn(
            f"Eigenvalues out of expected [0, 2] range: "
            f"min={eig_min:.4f}, max={eig_max:.4f}. "
            f"Check that L is the Laplacian and D is the degree matrix.",
            stacklevel=2
        )

    return U, eigenvalues, len(eigenvalues), ortho_error
# ============================================================================
# Training Functions
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
    import copy
    
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