"""
Utility functions for spectral basis experiments
Extended to support larger datasets: WikiCS, Amazon, Coauthor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

# Dataset loaders
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.datasets import Planetoid, WikiCS, Amazon, Coauthor
from torch_geometric.utils import to_undirected

# ============================================================================
# Model Architectures
# ============================================================================

class StandardMLP(nn.Module):
    """Standard MLP with bias and no normalization"""
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
        # Create standard 60/20/20 split
        print(f"Creating 60/20/20 split for {dataset_name}...")
        indices = np.arange(num_nodes)
        np.random.seed(42)
        np.random.shuffle(indices)
        
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
        # Create standard 60/20/20 split
        print(f"Creating 60/20/20 split for {dataset_name}...")
        indices = np.arange(num_nodes)
        np.random.seed(42)
        np.random.shuffle(indices)
        
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