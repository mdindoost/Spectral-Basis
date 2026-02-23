"""
Graph and dataset utilities for spectral basis experiments.

Covers:
  - Dataset loading (OGB, Planetoid, WikiCS, Amazon, Coauthor)
  - Graph matrix construction (adjacency, Laplacian, degree)
  - LCC extraction and subgraph operations
  - SGC diffusion preprocessing
  - Restricted eigenvector computation (Rayleigh-Ritz)
"""

import warnings
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import networkx as nx
import torch

from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.datasets import Planetoid, WikiCS, Amazon, Coauthor
from torch_geometric.utils import to_undirected

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
        warnings.warn(
            f"Eigenvalues out of expected [0, 2] range: "
            f"min={eig_min:.4f}, max={eig_max:.4f}. "
            f"Check that L is the Laplacian and D is the degree matrix.",
            stacklevel=2
        )

    return U, eigenvalues, len(eigenvalues), ortho_error
