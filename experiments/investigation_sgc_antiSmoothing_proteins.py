"""
Investigation: SGC Anti-Smoothing for ogbn-proteins (Multi-label)
==================================================================

Modified version for ogbn-proteins dataset which uses:
- Multi-label classification (132K nodes, 112 tasks)
- ROC-AUC metric instead of accuracy
- BCEWithLogitsLoss instead of CrossEntropyLoss

Usage:
    python experiments/investigation_sgc_antiSmoothing_proteins.py \
        --k_diffusion 2 4 6 8 10 \
        --splits fixed \
        --component lcc
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Fix for PyTorch 2.6 + OGB compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from utils import build_graph_matrices

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='SGC Anti-Smoothing for ogbn-proteins')
parser.add_argument('--k_diffusion', nargs='+', type=int, default=[2, 4, 6, 8, 10],
                   help='Diffusion propagation steps')
parser.add_argument('--splits', type=str, choices=['fixed'], default='fixed',
                   help='Use fixed splits (only option for ogbn-proteins)')
parser.add_argument('--component', type=str, choices=['whole', 'lcc'], default='lcc',
                   help='Use whole graph or largest connected component')
args = parser.parse_args()

DATASET_NAME = 'ogbn-proteins'
K_DIFFUSION_VALUES = args.k_diffusion
SPLIT_TYPE = 'fixed'  # Only fixed splits available
COMPONENT_TYPE = args.component

NUM_SEEDS = 5
EPOCHS = 200
HIDDEN_DIM = 256
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

output_base = f'results/investigation_sgc_antiSmoothing/{DATASET_NAME}_{SPLIT_TYPE}_{COMPONENT_TYPE}'
os.makedirs(f'{output_base}/plots', exist_ok=True)
os.makedirs(f'{output_base}/metrics', exist_ok=True)

print('='*80)
print('SGC ANTI-SMOOTHING: ogbn-proteins (Multi-label Classification)')
print('='*80)
print(f'Diffusion k values: {K_DIFFUSION_VALUES}')
print(f'Component: {COMPONENT_TYPE}')
print(f'Metric: ROC-AUC')
print(f'Device: {device}')
print('='*80)

# ============================================================================
# Multi-label Models
# ============================================================================

class SGC_MultiLabel(nn.Module):
    """SGC for multi-label classification"""
    def __init__(self, nfeat, ntasks):
        super(SGC_MultiLabel, self).__init__()
        self.W = nn.Linear(nfeat, ntasks, bias=True)
        torch.nn.init.xavier_normal_(self.W.weight)
    
    def forward(self, x):
        return self.W(x)  # Return logits (no sigmoid)

class StandardMLP_MultiLabel(nn.Module):
    """Standard MLP for multi-label"""
    def __init__(self, input_dim, hidden_dim, ntasks):
        super(StandardMLP_MultiLabel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, ntasks, bias=True)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Return logits

class RowNormMLP_MultiLabel(nn.Module):
    """RowNorm MLP for multi-label"""
    def __init__(self, input_dim, hidden_dim, ntasks):
        super(RowNormMLP_MultiLabel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, ntasks, bias=False)
    
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = F.relu(self.fc1(x))
        x = F.normalize(x, p=2, dim=1)
        x = F.relu(self.fc2(x))
        x = F.normalize(x, p=2, dim=1)
        return self.fc3(x)  # Return logits

# ============================================================================
# Helper Functions
# ============================================================================

def load_ogbn_proteins():
    """Load ogbn-proteins dataset"""
    from ogb.nodeproppred import NodePropPredDataset, Evaluator
    
    dataset = NodePropPredDataset(name='ogbn-proteins', root='./dataset')
    evaluator = Evaluator(name='ogbn-proteins')
    
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']
    
    graph, labels = dataset[0]
    
    edge_index = graph['edge_index']
    num_nodes = graph['num_nodes']
    
    # Convert labels to dense array (multi-label)
    labels = labels if isinstance(labels, np.ndarray) else labels.numpy()  # Shape: (num_nodes, num_tasks)
    num_tasks = labels.shape[1]
    
    # For node features: aggregate edge features to nodes
    # Strategy: mean of incoming edge features
    edge_feat = graph['edge_feat'] if isinstance(graph['edge_feat'], np.ndarray) else graph['edge_feat'].numpy()

    
    print(f'  Edge features: {edge_feat.shape}')
    print(f'  Aggregating to node features...')
    
    # Simple aggregation: for each node, average features of incident edges
    node_feat = np.zeros((num_nodes, edge_feat.shape[1]), dtype=np.float32)
    node_count = np.zeros(num_nodes, dtype=np.float32)
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        node_feat[dst] += edge_feat[i]
        node_count[dst] += 1
    
    # Average
    node_count[node_count == 0] = 1  # Avoid division by zero
    node_feat = node_feat / node_count[:, np.newaxis]
    
    print(f'  Node features created: {node_feat.shape}')
    
    return edge_index, node_feat, labels, num_nodes, num_tasks, train_idx, val_idx, test_idx, evaluator

def get_largest_connected_component(adj):
    """Extract LCC"""
    G = nx.from_scipy_sparse_array(adj)
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    lcc_nodes = sorted(list(largest_cc))
    
    lcc_mask = np.zeros(adj.shape[0], dtype=bool)
    lcc_mask[lcc_nodes] = True
    
    print(f'\n  Total components: {len(components)}')
    print(f'  Largest component: {len(largest_cc)} nodes ({len(largest_cc)/adj.shape[0]*100:.1f}%)')
    
    return lcc_mask

def extract_subgraph(adj, features, labels, mask, train_idx, val_idx, test_idx):
    """Extract subgraph"""
    node_indices = np.where(mask)[0]
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
    
    adj_sub = adj[mask][:, mask]
    features_sub = features[mask]
    labels_sub = labels[mask]
    
    # Remap indices
    train_idx_new = np.array([old_to_new[i] for i in train_idx if mask[i]])
    val_idx_new = np.array([old_to_new[i] for i in val_idx if mask[i]])
    test_idx_new = np.array([old_to_new[i] for i in test_idx if mask[i]])
    
    print(f'  Train: {len(train_idx)} -> {len(train_idx_new)}')
    print(f'  Val: {len(val_idx)} -> {len(val_idx_new)}')
    print(f'  Test: {len(test_idx)} -> {len(test_idx_new)}')
    
    return adj_sub, features_sub, labels_sub, train_idx_new, val_idx_new, test_idx_new

def compute_sgc_normalized_adjacency(adj):
    """SGC normalization"""
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def sgc_precompute(features, adj_normalized, degree):
    """SGC diffusion"""
    for i in range(degree):
        features = adj_normalized @ features
    return features

def compute_restricted_eigenvectors(X, L, D, num_components=0):
    """Compute restricted eigenvectors"""
    num_nodes, dimension = X.shape
    
    Q, R = np.linalg.qr(X)
    rank_X = np.sum(np.abs(np.diag(R)) > 1e-10)
    
    if rank_X < dimension:
        Q = Q[:, :rank_X]
        dimension = rank_X
    
    L_r = Q.T @ (L @ Q)
    D_r = Q.T @ (D @ Q)
    
    L_r = 0.5 * (L_r + L_r.T)
    D_r = 0.5 * (D_r + D_r.T)
    
    eps = 1e-10 * np.trace(D_r) / dimension
    D_r = D_r + eps * np.eye(dimension)
    
    eigenvalues, V = la.eigh(L_r, D_r)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    if num_components > 0:
        eigenvalues = eigenvalues[num_components:]
        V = V[:, num_components:]
    
    U = Q @ V
    
    DU = D @ U
    G = U.T @ DU
    ortho_error = np.abs(G - np.eye(U.shape[1])).max()
    
    return U.astype(np.float32), eigenvalues, U.shape[1], ortho_error

def random_binary_projection(Y, target_dim, seed=42):
    """Binary JL projection"""
    np.random.seed(seed)
    H = np.random.choice([1, -1], size=(Y.shape[1], target_dim)).astype(np.float32)
    H = H / np.sqrt(target_dim)
    return Y @ H

def random_orthogonal_projection(Y, target_dim, seed=42):
    """Orthogonal JL projection"""
    np.random.seed(seed)
    H = np.random.randn(Y.shape[1], target_dim).astype(np.float32)
    H, _ = np.linalg.qr(H)
    return Y @ H

def compute_rocauc(y_true, y_pred):
    """
    Compute ROC-AUC for multi-label classification
    
    Args:
        y_true: (n, num_tasks) binary labels
        y_pred: (n, num_tasks) predicted probabilities
    
    Returns:
        rocauc: Average ROC-AUC across tasks
    """
    # Remove NaN labels
    valid_mask = ~np.isnan(y_true)
    
    rocauc_list = []
    for i in range(y_true.shape[1]):
        task_valid = valid_mask[:, i]
        if task_valid.sum() > 0 and len(np.unique(y_true[task_valid, i])) > 1:
            rocauc = roc_auc_score(y_true[task_valid, i], y_pred[task_valid, i])
            rocauc_list.append(rocauc)
    
    if len(rocauc_list) == 0:
        return 0.0
    
    return np.mean(rocauc_list)

def aggregate_results(results):
    """Aggregate results"""
    val_aucs = [r['val_auc'] for r in results]
    test_aucs = [r['test_auc'] for r in results]
    train_times = [r['train_time'] for r in results]
    
    return {
        'val_auc_mean': float(np.mean(val_aucs)),
        'val_auc_std': float(np.std(val_aucs)),
        'test_auc_mean': float(np.mean(test_aucs)),
        'test_auc_std': float(np.std(test_aucs)),
        'train_time_mean': float(np.mean(train_times)),
        'train_time_std': float(np.std(train_times))
    }

def train_and_test_multilabel(model, X_train, y_train, X_val, y_val, X_test, y_test,
                               epochs, lr, weight_decay, device):
    """Train and test multi-label model"""
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()  # Multi-label loss
    
    start_time = time.time()
    best_val_auc = 0.0
    best_test_auc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        
        # Mask NaN labels
        train_mask = ~torch.isnan(y_train_t)
        loss = criterion(output[train_mask], y_train_t[train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Evaluate
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_output = torch.sigmoid(model(X_val_t))
                test_output = torch.sigmoid(model(X_test_t))
                
                val_auc = compute_rocauc(y_val, val_output.cpu().numpy())
                test_auc = compute_rocauc(y_test, test_output.cpu().numpy())
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_test_auc = test_auc
    
    train_time = time.time() - start_time
    return best_val_auc, best_test_auc, train_time

# ============================================================================
# Experiment Functions
# ============================================================================

def run_sgc_baseline(X_diffused, labels, train_idx, val_idx, test_idx, num_tasks, num_seeds, device):
    """SGC Baseline"""
    results = []
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = SGC_MultiLabel(X_diffused.shape[1], num_tasks).to(device)
        
        val_auc, test_auc, train_time = train_and_test_multilabel(
            model, X_diffused[train_idx], labels[train_idx],
            X_diffused[val_idx], labels[val_idx],
            X_diffused[test_idx], labels[test_idx],
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device
        )
        
        results.append({
            'seed': seed,
            'val_auc': val_auc,
            'test_auc': test_auc,
            'train_time': train_time
        })
    
    return results

def run_sgc_mlp_baseline(X_diffused, labels, train_idx, val_idx, test_idx, num_tasks, num_seeds, device):
    """SGC + MLP Baseline"""
    results = []
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_diffused[train_idx])
    X_val_scaled = scaler.transform(X_diffused[val_idx])
    X_test_scaled = scaler.transform(X_diffused[test_idx])
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = StandardMLP_MultiLabel(X_diffused.shape[1], HIDDEN_DIM, num_tasks).to(device)
        
        val_auc, test_auc, train_time = train_and_test_multilabel(
            model, X_scaled, labels[train_idx],
            X_val_scaled, labels[val_idx],
            X_test_scaled, labels[test_idx],
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device
        )
        
        results.append({
            'seed': seed,
            'val_auc': val_auc,
            'test_auc': test_auc,
            'train_time': train_time
        })
    
    return results

def run_restricted_eigenvectors(X_diffused, L, D, num_components, labels, 
                                train_idx, val_idx, test_idx, num_tasks, num_seeds, device):
    """Full Restricted Eigenvectors + RowNorm"""
    U, eigenvalues, d_eff, ortho_err = compute_restricted_eigenvectors(X_diffused, L, D, num_components)
    
    results = []
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = RowNormMLP_MultiLabel(U.shape[1], HIDDEN_DIM, num_tasks).to(device)
        
        val_auc, test_auc, train_time = train_and_test_multilabel(
            model, U[train_idx], labels[train_idx],
            U[val_idx], labels[val_idx],
            U[test_idx], labels[test_idx],
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, device
        )
        
        results.append({
            'seed': seed,
            'val_auc': val_auc,
            'test_auc': test_auc,
            'train_time': train_time
        })
    
    return results, d_eff, ortho_err

# ============================================================================
# Main Experiment
# ============================================================================

print('\n[1/6] Loading ogbn-proteins dataset...')
(edge_index, features, labels, num_nodes, num_tasks, 
 train_idx, val_idx, test_idx, evaluator) = load_ogbn_proteins()

print(f'  Nodes: {num_nodes:,}')
print(f'  Features: {features.shape}')
print(f'  Tasks (multi-label): {num_tasks}')
print(f'  Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}')

print('\n[2/6] Building graph matrices...')
adj, D, L = build_graph_matrices(edge_index, num_nodes)
print(f'  Adjacency: {adj.shape}')

if COMPONENT_TYPE == 'lcc':
    print('\n[3/6] Extracting LCC...')
    lcc_mask = get_largest_connected_component(adj)
    adj, features, labels, train_idx, val_idx, test_idx = extract_subgraph(
        adj, features, labels, lcc_mask, train_idx, val_idx, test_idx
    )
    
    print('  Rebuilding graph matrices for LCC...')
    adj_coo = adj.tocoo()
    edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
    adj, D, L = build_graph_matrices(edge_index_lcc, adj.shape[0])
else:
    print('\n[3/6] Using whole graph...')

num_nodes = features.shape[0]

G = nx.from_scipy_sparse_array(adj)
num_components = nx.number_connected_components(G)
print(f'\n[4/6] Connected components: {num_components}')

all_results = {}

for k_diff in K_DIFFUSION_VALUES:
    print(f'\n{"="*80}')
    print(f'DIFFUSION k={k_diff}')
    print(f'{"="*80}')
    
    output_k = f'{output_base}/k{k_diff}'
    os.makedirs(f'{output_k}/metrics', exist_ok=True)
    
    print('\n[5/6] Computing SGC diffusion...')
    A_sgc = compute_sgc_normalized_adjacency(adj)
    
    start_time = time.time()
    X_diffused = sgc_precompute(features.copy(), A_sgc, k_diff)
    precompute_time = time.time() - start_time
    print(f'  ✓ Done in {precompute_time:.2f}s')
    
    experiments = {
        'sgc_baseline': [],
        'sgc_mlp_baseline': [],
        'full_rownorm_mlp': [],
    }
    
    print(f'\n[6/6] Running experiments...')
    
    # SGC
    print('\n[1/3] SGC Baseline...')
    sgc_results = run_sgc_baseline(X_diffused, labels, train_idx, val_idx, test_idx, num_tasks, NUM_SEEDS, device)
    experiments['sgc_baseline'] = sgc_results
    sgc_agg = aggregate_results(sgc_results)
    print(f'  → {sgc_agg["test_auc_mean"]*100:.2f}% ± {sgc_agg["test_auc_std"]*100:.2f}%')
    
    # SGC + MLP
    print('\n[2/3] SGC + MLP...')
    sgc_mlp_results = run_sgc_mlp_baseline(X_diffused, labels, train_idx, val_idx, test_idx, num_tasks, NUM_SEEDS, device)
    experiments['sgc_mlp_baseline'] = sgc_mlp_results
    sgc_mlp_agg = aggregate_results(sgc_mlp_results)
    print(f'  → {sgc_mlp_agg["test_auc_mean"]*100:.2f}% ± {sgc_mlp_agg["test_auc_std"]*100:.2f}%')
    
    # Full + RowNorm
    print('\n[3/3] Full + RowNorm MLP...')
    full_results, d_eff, ortho_err = run_restricted_eigenvectors(
        X_diffused, L, D, num_components, labels, train_idx, val_idx, test_idx, num_tasks, NUM_SEEDS, device
    )
    experiments['full_rownorm_mlp'] = full_results
    full_agg = aggregate_results(full_results)
    print(f'  → {full_agg["test_auc_mean"]*100:.2f}% ± {full_agg["test_auc_std"]*100:.2f}%')
    print(f'  D-orthonormality: {ortho_err:.2e}')
    
    # Save
    final_results = {k: aggregate_results(v) for k, v in experiments.items()}
    
    results_dict = {
        'dataset': DATASET_NAME,
        'k_diffusion': k_diff,
        'num_seeds': NUM_SEEDS,
        'metric': 'ROC-AUC',
        'results': final_results
    }
    
    save_path = f'{output_k}/metrics/results.json'
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    all_results[k_diff] = results_dict
    print(f'\n✓ Saved: {save_path}')

print(f'\n{"="*80}')
print('EXPERIMENT COMPLETE')
print(f'{"="*80}')
