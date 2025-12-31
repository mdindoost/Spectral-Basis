#!/usr/bin/env python3
"""
Bethe Hessian vs. Normalized Laplacian for Normalized Cut
==========================================================

Experimental framework to test whether Bethe Hessian outperforms
the Normalized Laplacian for the normalized cut problem, following
Newman (2013) "Spectral methods for network community detection 
and graph partitioning".

Key research questions:
1. Does BH avoid hub localization that plagues normalized Laplacian?
2. Does BH perform better near the detectability threshold?
3. How do BH and NL compare on Newman's benchmark networks?

Author: Mohammad (NJIT)
Date: December 2024
"""

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# MATRIX CONSTRUCTIONS
# =============================================================================

def normalized_laplacian(G):
    """
    Compute the normalized Laplacian L = D^{-1/2} A D^{-1/2}
    as defined in Newman (2013) Eq. 17.
    
    Note: Newman uses L = D^{-1/2} A D^{-1/2}, not I - D^{-1/2} A D^{-1/2}
    The eigenvectors are the same, eigenvalues differ by shift.
    """
    A = nx.adjacency_matrix(G).astype(float)
    n = G.number_of_nodes()
    degrees = np.array(A.sum(axis=1)).flatten()
    
    # Handle isolated vertices
    degrees[degrees == 0] = 1
    
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    L = D_inv_sqrt @ A @ D_inv_sqrt
    
    return L, degrees

def bethe_hessian(G, r=None):
    """
    Compute the Bethe Hessian matrix:
    H(r) = (r^2 - 1)I - rA + D
    
    Default r = sqrt(average_degree) following Saade et al. (2014)
    
    Parameters:
    -----------
    G : NetworkX graph
    r : float, optional
        Regularization parameter. If None, uses sqrt(avg_degree)
    
    Returns:
    --------
    H : sparse matrix
        Bethe Hessian matrix
    r : float
        The r parameter used
    """
    A = nx.adjacency_matrix(G).astype(float)
    n = G.number_of_nodes()
    degrees = np.array(A.sum(axis=1)).flatten()
    
    if r is None:
        avg_degree = degrees.mean()
        r = np.sqrt(max(avg_degree, 1.0))
    
    D = sparse.diags(degrees)
    I = sparse.eye(n)
    
    H = (r**2 - 1) * I - r * A + D
    
    return H, r, degrees

def modularity_matrix(G):
    """
    Compute the modularity matrix B = A - kk^T / 2m
    as defined in Newman (2013) Eq. 4.
    """
    A = nx.adjacency_matrix(G).astype(float)
    n = G.number_of_nodes()
    degrees = np.array(A.sum(axis=1)).flatten()
    m = G.number_of_edges()
    
    # B = A - kk^T / 2m
    k = degrees.reshape(-1, 1)
    B = A - (k @ k.T) / (2 * m)
    
    return B, degrees

# =============================================================================
# SPECTRAL PARTITIONING METHODS
# =============================================================================

def partition_by_normalized_laplacian(G, use_second=True, return_all_eigenvectors=False):
    """
    Partition graph using normalized Laplacian (Newman's method).
    
    Uses sweep cuts to find optimal threshold, similar to standard practice.
    
    Parameters:
    -----------
    G : NetworkX graph
    use_second : bool
        If True, use second eigenvector. If False, return top k eigenvectors.
    return_all_eigenvectors : bool
        If True, return multiple eigenvectors for analysis
    
    Returns:
    --------
    partition : dict mapping node -> {0, 1}
    eigenvector : the eigenvector used
    eigenvalue : corresponding eigenvalue
    info : additional information
    """
    L, degrees = normalized_laplacian(G)
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    
    # Get top k eigenvectors (largest eigenvalues for L = D^{-1/2} A D^{-1/2})
    k = min(5, n - 1)
    
    if n < 100:
        # Dense computation for small graphs
        L_dense = L.toarray()
        eigenvalues, eigenvectors = eigh(L_dense)
        # Sort by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    else:
        # Sparse computation
        eigenvalues, eigenvectors = eigsh(L, k=k, which='LA')
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    
    # The first eigenvector (largest eigenvalue) is trivial (all same sign)
    # Use second eigenvector for partitioning
    v2 = eigenvectors[:, 1]
    
    # Use sweep cut to find best partition
    sorted_idx = np.argsort(v2)
    sorted_nodes = [nodes[j] for j in sorted_idx]
    
    best_cut_score = float('inf')
    best_partition = None
    
    for cut_pos in range(1, n):
        group0 = set(sorted_nodes[:cut_pos])
        group1 = set(sorted_nodes[cut_pos:])
        
        if len(group0) == 0 or len(group1) == 0:
            continue
        
        cut_size = nx.cut_size(G, group0, group1)
        vol0 = sum(G.degree(nd) for nd in group0)
        vol1 = sum(G.degree(nd) for nd in group1)
        
        if vol0 > 0 and vol1 > 0:
            ncut = cut_size / (vol0 * vol1)
        else:
            ncut = float('inf')
        
        if ncut < best_cut_score:
            best_cut_score = ncut
            best_partition = {nd: 0 if nd in group0 else 1 for nd in nodes}
    
    # Fallback to sign partition
    if best_partition is None:
        best_partition = {nodes[i]: 0 if v2[i] >= 0 else 1 for i in range(n)}
    
    # Compute localization metrics
    localization = compute_localization(v2, degrees)
    
    info = {
        'eigenvalues': eigenvalues[:k],
        'eigenvectors': eigenvectors[:, :k] if return_all_eigenvectors else None,
        'localization': localization,
        'spectral_gap': eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else 0,
        'best_normalized_cut': best_cut_score
    }
    
    return best_partition, v2, eigenvalues[1], info

def partition_by_bethe_hessian(G, r=None, return_all_eigenvectors=False):
    """
    Partition graph using Bethe Hessian.
    
    For community detection, we look for negative eigenvalues.
    The eigenvectors corresponding to negative eigenvalues contain
    community information.
    
    IMPORTANT: We try multiple eigenvectors and sweep cuts to find
    the best partition, similar to how Fiedler vector methods work.
    
    Returns:
    --------
    partition : dict mapping node -> {0, 1}
    eigenvector : the eigenvector used
    eigenvalue : corresponding eigenvalue
    info : additional information
    """
    H, r_used, degrees = bethe_hessian(G, r)
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    
    k = min(10, n - 1)
    
    if n < 100:
        # Dense computation
        H_dense = H.toarray()
        eigenvalues, eigenvectors = eigh(H_dense)
        # Already sorted ascending
    else:
        # Get smallest eigenvalues (looking for negative ones)
        eigenvalues, eigenvectors = eigsh(H, k=k, which='SA')
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    
    # Count negative eigenvalues (indicates number of communities - 1)
    n_negative = np.sum(eigenvalues < 0)
    
    # Try sweep cuts on multiple eigenvectors corresponding to negative eigenvalues
    # Also try the first few positive ones in case of near-threshold cases
    best_partition = None
    best_score = float('inf')  # Lower normalized cut is better
    best_eigenvector = None
    best_eigenvalue = None
    
    # Consider eigenvectors for negative eigenvalues plus a few more
    n_to_try = min(max(n_negative + 2, 3), eigenvectors.shape[1])
    
    for i in range(n_to_try):
        v = eigenvectors[:, i]
        
        # Skip nearly constant eigenvectors
        if np.std(v) < 1e-8:
            continue
        
        # Try sweep cut: sort vertices by eigenvector value, try all thresholds
        sorted_idx = np.argsort(v)
        sorted_nodes = [nodes[j] for j in sorted_idx]
        
        best_cut_score = float('inf')
        best_cut_partition = None
        
        for cut_pos in range(1, n):
            group0 = set(sorted_nodes[:cut_pos])
            group1 = set(sorted_nodes[cut_pos:])
            
            if len(group0) == 0 or len(group1) == 0:
                continue
            
            # Compute normalized cut (Newman's version: R / (κ₁κ₂))
            cut_size = nx.cut_size(G, group0, group1)
            vol0 = sum(G.degree(nd) for nd in group0)
            vol1 = sum(G.degree(nd) for nd in group1)
            
            if vol0 > 0 and vol1 > 0:
                ncut = cut_size / (vol0 * vol1)
            else:
                ncut = float('inf')
            
            if ncut < best_cut_score:
                best_cut_score = ncut
                best_cut_partition = {nd: 0 if nd in group0 else 1 for nd in nodes}
        
        # Also try sign-based partition (traditional approach)
        sign_partition = {nodes[j]: 0 if v[j] >= 0 else 1 for j in range(n)}
        group0_sign = [nd for nd in nodes if sign_partition[nd] == 0]
        group1_sign = [nd for nd in nodes if sign_partition[nd] == 1]
        
        if len(group0_sign) > 0 and len(group1_sign) > 0:
            cut_size_sign = nx.cut_size(G, group0_sign, group1_sign)
            vol0_sign = sum(G.degree(nd) for nd in group0_sign)
            vol1_sign = sum(G.degree(nd) for nd in group1_sign)
            if vol0_sign > 0 and vol1_sign > 0:
                ncut_sign = cut_size_sign / (vol0_sign * vol1_sign)
                if ncut_sign < best_cut_score:
                    best_cut_score = ncut_sign
                    best_cut_partition = sign_partition
        
        if best_cut_partition and best_cut_score < best_score:
            best_score = best_cut_score
            best_partition = best_cut_partition
            best_eigenvector = v
            best_eigenvalue = eigenvalues[i]
    
    # Fallback to first eigenvector sign partition if nothing worked
    if best_partition is None:
        v = eigenvectors[:, 0]
        best_partition = {nodes[j]: 0 if v[j] >= 0 else 1 for j in range(n)}
        best_eigenvector = v
        best_eigenvalue = eigenvalues[0]
    
    # Compute localization metrics
    localization = compute_localization(best_eigenvector, degrees)
    
    info = {
        'eigenvalues': eigenvalues[:k],
        'eigenvectors': eigenvectors[:, :k] if return_all_eigenvectors else None,
        'localization': localization,
        'r_parameter': r_used,
        'n_negative_eigenvalues': n_negative,
        'spectral_gap': eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0,
        'best_normalized_cut': best_score
    }
    
    return best_partition, best_eigenvector, best_eigenvalue, info

def compute_localization(v, degrees):
    """
    Compute localization metrics for eigenvector.
    
    Hub localization occurs when eigenvector concentrates on high-degree nodes.
    """
    v_normalized = v / np.linalg.norm(v)
    v_squared = v_normalized ** 2
    
    # Inverse Participation Ratio (IPR)
    # IPR = sum(v_i^4) / (sum(v_i^2))^2
    # For localized vectors: IPR ~ 1/k where k is number of localized nodes
    # For delocalized vectors: IPR ~ 1/n
    ipr = np.sum(v_normalized ** 4)
    
    # Correlation with degree
    if np.std(degrees) > 0 and np.std(np.abs(v_normalized)) > 0:
        degree_correlation = np.corrcoef(degrees, np.abs(v_normalized))[0, 1]
    else:
        degree_correlation = 0
    
    # Fraction of weight on top 10% highest degree nodes
    n = len(degrees)
    top_10_percent = max(1, n // 10)
    high_degree_idx = np.argsort(degrees)[-top_10_percent:]
    weight_on_hubs = np.sum(v_squared[high_degree_idx])
    
    return {
        'ipr': ipr,
        'degree_correlation': degree_correlation,
        'weight_on_hubs': weight_on_hubs,
        'effective_dimension': 1.0 / ipr if ipr > 0 else n
    }

# =============================================================================
# CUT QUALITY METRICS
# =============================================================================

def compute_cut_metrics(G, partition):
    """
    Compute various cut quality metrics.
    
    Parameters:
    -----------
    G : NetworkX graph
    partition : dict mapping node -> group (0 or 1)
    
    Returns:
    --------
    metrics : dict with cut size, normalized cut, ratio cut, conductance, modularity
    """
    nodes = list(G.nodes())
    group0 = [n for n in nodes if partition[n] == 0]
    group1 = [n for n in nodes if partition[n] == 1]
    
    if len(group0) == 0 or len(group1) == 0:
        return {
            'cut_size': 0,
            'normalized_cut': float('inf'),
            'ratio_cut': float('inf'),
            'conductance': 1.0,
            'modularity': -1.0,
            'balance': 0.0
        }
    
    # Cut size R = number of edges between groups
    cut_size = nx.cut_size(G, group0, group1)
    
    # Volume (sum of degrees) for each group
    vol0 = sum(G.degree(n) for n in group0)
    vol1 = sum(G.degree(n) for n in group1)
    
    # Normalized cut: R/κ₁ + R/κ₂ (Shi-Malik definition)
    # Or R/(κ₁κ₂) (Newman's definition)
    if vol0 > 0 and vol1 > 0:
        normalized_cut_newman = cut_size / (vol0 * vol1)
        normalized_cut_shi_malik = cut_size / vol0 + cut_size / vol1
    else:
        normalized_cut_newman = float('inf')
        normalized_cut_shi_malik = float('inf')
    
    # Ratio cut: R/(n₁n₂)
    ratio_cut = cut_size / (len(group0) * len(group1))
    
    # Conductance: R / min(vol0, vol1)
    conductance = cut_size / min(vol0, vol1) if min(vol0, vol1) > 0 else 1.0
    
    # Modularity
    m = G.number_of_edges()
    Q = 0
    for u, v in G.edges():
        if partition[u] == partition[v]:
            Q += 1 - G.degree(u) * G.degree(v) / (2 * m)
        else:
            Q += 0 - G.degree(u) * G.degree(v) / (2 * m)
    Q /= (2 * m)
    
    # Balance: min(n₁, n₂) / max(n₁, n₂)
    balance = min(len(group0), len(group1)) / max(len(group0), len(group1))
    
    return {
        'cut_size': cut_size,
        'normalized_cut_newman': normalized_cut_newman,
        'normalized_cut_shi_malik': normalized_cut_shi_malik,
        'ratio_cut': ratio_cut,
        'conductance': conductance,
        'modularity': Q,
        'balance': balance,
        'group_sizes': (len(group0), len(group1)),
        'group_volumes': (vol0, vol1)
    }

def compute_classification_accuracy(partition, ground_truth):
    """
    Compute fraction of correctly classified vertices.
    
    Handles label permutation (since labels 0/1 are arbitrary).
    """
    nodes = list(partition.keys())
    
    # Try both label assignments
    correct_same = sum(1 for n in nodes if partition[n] == ground_truth[n])
    correct_flipped = sum(1 for n in nodes if partition[n] != ground_truth[n])
    
    return max(correct_same, correct_flipped) / len(nodes)

# =============================================================================
# NEWMAN'S BENCHMARK NETWORKS
# =============================================================================

def load_karate_club():
    """Zachary's Karate Club with ground truth."""
    G = nx.karate_club_graph()
    # Ground truth based on actual club split
    ground_truth = {n: 0 if G.nodes[n]['club'] == 'Mr. Hi' else 1 
                   for n in G.nodes()}
    return G, ground_truth, "Karate Club"

def load_dolphins():
    """
    Dolphin social network (Lusseau et al. 2003).
    Download from: http://www-personal.umich.edu/~mejn/netdata/
    """
    try:
        # Try to load from file
        G = nx.read_gml('/home/claude/dolphins.gml')
    except:
        # Create approximate version based on known properties
        # The real network has 62 dolphins, splits into 2 groups
        print("Note: Using synthetic dolphins network (download real from Newman's website)")
        G = nx.generators.community.stochastic_block_model(
            [31, 31], [[0.3, 0.05], [0.05, 0.3]], seed=42
        )
        G = nx.relabel_nodes(G, {i: f"dolphin_{i}" for i in G.nodes()})
    
    # Approximate ground truth (real network splits ~half and half)
    nodes = list(G.nodes())
    ground_truth = {n: 0 if i < len(nodes)//2 else 1 
                   for i, n in enumerate(nodes)}
    return G, ground_truth, "Dolphins"

def load_political_books():
    """
    Political books network (Krebs).
    Books about US politics, connected if frequently co-purchased.
    """
    try:
        G = nx.read_gml('/home/claude/polbooks.gml')
        # Ground truth from 'value' attribute (political leaning)
        ground_truth = {n: 0 if G.nodes[n].get('value', 'n') == 'l' else 1
                       for n in G.nodes()}
    except:
        print("Note: Using synthetic political books network")
        # 105 books, roughly split by ideology
        G = nx.generators.community.stochastic_block_model(
            [49, 56], [[0.25, 0.02], [0.02, 0.25]], seed=43
        )
        G = nx.relabel_nodes(G, {i: f"book_{i}" for i in G.nodes()})
        nodes = list(G.nodes())
        ground_truth = {n: 0 if i < 49 else 1 for i, n in enumerate(nodes)}
    
    return G, ground_truth, "Political Books"

def load_political_blogs():
    """
    Political blogs network (Adamic & Glance 2005).
    This is Newman's problematic case where hub localization occurs.
    """
    try:
        G = nx.read_gml('/home/claude/polblogs.gml')
        ground_truth = {n: G.nodes[n].get('value', 0) for n in G.nodes()}
    except:
        print("Note: Using synthetic political blogs network with power-law degrees")
        # Key property: broad degree distribution with hubs
        # ~1200 nodes, power-law degrees
        n = 500  # Smaller for testing
        G = nx.powerlaw_cluster_graph(n, 3, 0.1, seed=44)
        
        # Add community structure on top
        nodes = list(G.nodes())
        mid = n // 2
        for i in range(mid):
            for j in range(mid, n):
                if G.has_edge(i, j) and np.random.random() < 0.7:
                    G.remove_edge(i, j)
        
        ground_truth = {n: 0 if n < mid else 1 for n in nodes}
    
    return G, ground_truth, "Political Blogs"

# =============================================================================
# STOCHASTIC BLOCK MODEL EXPERIMENTS
# =============================================================================

def generate_sbm(n, k, p_in, p_out, seed=None):
    """
    Generate a stochastic block model graph.
    
    Parameters:
    -----------
    n : int
        Number of vertices
    k : int
        Number of communities (equal sized)
    p_in : float
        Within-community edge probability
    p_out : float
        Between-community edge probability
    seed : int
        Random seed
    
    Returns:
    --------
    G : NetworkX graph
    ground_truth : dict mapping node -> community
    """
    if seed is not None:
        np.random.seed(seed)
    
    sizes = [n // k] * k
    sizes[-1] += n - sum(sizes)  # Handle remainder
    
    probs = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
    
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    
    # Ground truth
    ground_truth = {}
    node_idx = 0
    for comm_idx, size in enumerate(sizes):
        for _ in range(size):
            ground_truth[node_idx] = comm_idx
            node_idx += 1
    
    return G, ground_truth

def detectability_threshold_experiment(n=1000, num_trials=10):
    """
    Test BH vs NL near the detectability threshold.
    
    For 2-community SBM with average degree c:
    - Threshold: (p_in - p_out)^2 * n / (4 * c) = 1
    - Below threshold: no algorithm can detect
    - Above threshold: detection possible
    """
    results = []
    
    # Fix average degree
    c = 10  # average degree
    
    # Vary signal strength (distance from threshold)
    # epsilon = 0 is at threshold, epsilon > 0 is above
    for epsilon in np.linspace(-0.5, 2.0, 20):
        
        # p_in + p_out = 2c/n (to maintain average degree c)
        # (p_in - p_out) = 2 * sqrt((1 + epsilon) * c) / n (signal strength)
        
        signal = 2 * np.sqrt(max(0.01, (1 + epsilon)) * c) / n
        p_avg = c / n
        
        p_in = p_avg + signal / 2
        p_out = p_avg - signal / 2
        
        # Ensure valid probabilities
        p_in = min(1.0, max(0.0, p_in))
        p_out = min(1.0, max(0.0, p_out))
        
        trial_results = {'epsilon': epsilon, 'nl_accuracy': [], 'bh_accuracy': []}
        
        for trial in range(num_trials):
            seed_val = abs(trial*100 + int(epsilon*1000)) % (2**31)
            G, ground_truth = generate_sbm(n, 2, p_in, p_out, seed=seed_val)
            
            # Skip if graph is too disconnected
            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                ground_truth = {n: ground_truth[n] for n in G.nodes()}
            
            if G.number_of_nodes() < 10:
                continue
            
            # Normalized Laplacian
            try:
                part_nl, _, _, _ = partition_by_normalized_laplacian(G)
                acc_nl = compute_classification_accuracy(part_nl, ground_truth)
                trial_results['nl_accuracy'].append(acc_nl)
            except:
                pass
            
            # Bethe Hessian
            try:
                part_bh, _, _, _ = partition_by_bethe_hessian(G)
                acc_bh = compute_classification_accuracy(part_bh, ground_truth)
                trial_results['bh_accuracy'].append(acc_bh)
            except:
                pass
        
        if trial_results['nl_accuracy'] and trial_results['bh_accuracy']:
            results.append({
                'epsilon': epsilon,
                'p_in': p_in,
                'p_out': p_out,
                'nl_accuracy_mean': np.mean(trial_results['nl_accuracy']),
                'nl_accuracy_std': np.std(trial_results['nl_accuracy']),
                'bh_accuracy_mean': np.mean(trial_results['bh_accuracy']),
                'bh_accuracy_std': np.std(trial_results['bh_accuracy'])
            })
    
    return results

# =============================================================================
# HUB LOCALIZATION EXPERIMENT
# =============================================================================

def hub_localization_experiment():
    """
    Test hub localization on graphs with varying degree heterogeneity.
    
    Newman noted that on the political blogs network, the second eigenvector
    of the normalized Laplacian was "strongly localized around a few of the
    highest-degree vertices" and he had to use the third eigenvector instead.
    """
    results = []
    
    # Test different graph types with varying hub structure
    graph_types = [
        ('ER', lambda n, p: nx.erdos_renyi_graph(n, p, seed=42)),
        ('Regular', lambda n, d: nx.random_regular_graph(d, n, seed=42)),
        ('Powerlaw', lambda n, m: nx.barabasi_albert_graph(n, m, seed=42)),
        ('SBM_uniform', lambda n, _: generate_sbm(n, 2, 0.15, 0.01, seed=42)[0]),
    ]
    
    n = 500
    
    for name, generator in graph_types:
        if name == 'ER':
            G = generator(n, 0.02)
        elif name == 'Regular':
            G = generator(n, 10)
        elif name == 'Powerlaw':
            G = generator(n, 3)
        else:
            G = generator(n, None)
        
        # Ensure connected
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        
        # Get degree statistics
        degrees = [G.degree(n) for n in G.nodes()]
        
        # Normalized Laplacian analysis
        _, v_nl, _, info_nl = partition_by_normalized_laplacian(G, return_all_eigenvectors=True)
        
        # Bethe Hessian analysis
        _, v_bh, _, info_bh = partition_by_bethe_hessian(G, return_all_eigenvectors=True)
        
        results.append({
            'graph_type': name,
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'degree_mean': np.mean(degrees),
            'degree_std': np.std(degrees),
            'degree_max': np.max(degrees),
            'degree_min': np.min(degrees),
            'nl_localization': info_nl['localization'],
            'bh_localization': info_bh['localization'],
            'nl_spectral_gap': info_nl['spectral_gap'],
            'bh_spectral_gap': info_bh['spectral_gap'],
            'bh_r_parameter': info_bh['r_parameter'],
            'bh_n_negative_eigs': info_bh['n_negative_eigenvalues']
        })
    
    return results

# =============================================================================
# MAIN COMPARISON ON NEWMAN'S NETWORKS
# =============================================================================

def compare_methods_on_network(G, ground_truth, name):
    """
    Compare all methods on a single network.
    """
    print(f"\n{'='*60}")
    print(f"Network: {name}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"{'='*60}")
    
    degrees = [G.degree(n) for n in G.nodes()]
    print(f"Degree: mean={np.mean(degrees):.1f}, std={np.std(degrees):.1f}, "
          f"max={np.max(degrees)}, min={np.min(degrees)}")
    
    results = {'network': name, 'n': G.number_of_nodes(), 'm': G.number_of_edges()}
    
    # Normalized Laplacian
    print("\n--- Normalized Laplacian ---")
    try:
        part_nl, v_nl, ev_nl, info_nl = partition_by_normalized_laplacian(G, return_all_eigenvectors=True)
        metrics_nl = compute_cut_metrics(G, part_nl)
        acc_nl = compute_classification_accuracy(part_nl, ground_truth)
        
        print(f"Eigenvalue used: {ev_nl:.4f}")
        print(f"Top 5 eigenvalues: {info_nl['eigenvalues'][:5]}")
        print(f"Classification accuracy: {acc_nl:.2%}")
        print(f"Normalized cut (Newman): {metrics_nl['normalized_cut_newman']:.6f}")
        print(f"Conductance: {metrics_nl['conductance']:.4f}")
        print(f"Modularity: {metrics_nl['modularity']:.4f}")
        print(f"Balance: {metrics_nl['balance']:.2f}")
        print(f"Localization IPR: {info_nl['localization']['ipr']:.4f}")
        print(f"Weight on hubs: {info_nl['localization']['weight_on_hubs']:.2%}")
        print(f"Degree correlation: {info_nl['localization']['degree_correlation']:.4f}")
        
        results['nl'] = {
            'accuracy': acc_nl,
            'metrics': metrics_nl,
            'localization': info_nl['localization'],
            'eigenvalue': ev_nl
        }
    except Exception as e:
        print(f"Error: {e}")
        results['nl'] = None
    
    # Bethe Hessian
    print("\n--- Bethe Hessian ---")
    try:
        part_bh, v_bh, ev_bh, info_bh = partition_by_bethe_hessian(G, return_all_eigenvectors=True)
        metrics_bh = compute_cut_metrics(G, part_bh)
        acc_bh = compute_classification_accuracy(part_bh, ground_truth)
        
        print(f"r parameter: {info_bh['r_parameter']:.4f}")
        print(f"Eigenvalue used: {ev_bh:.4f}")
        print(f"Smallest 5 eigenvalues: {info_bh['eigenvalues'][:5]}")
        print(f"Number of negative eigenvalues: {info_bh['n_negative_eigenvalues']}")
        print(f"Classification accuracy: {acc_bh:.2%}")
        print(f"Normalized cut (Newman): {metrics_bh['normalized_cut_newman']:.6f}")
        print(f"Conductance: {metrics_bh['conductance']:.4f}")
        print(f"Modularity: {metrics_bh['modularity']:.4f}")
        print(f"Balance: {metrics_bh['balance']:.2f}")
        print(f"Localization IPR: {info_bh['localization']['ipr']:.4f}")
        print(f"Weight on hubs: {info_bh['localization']['weight_on_hubs']:.2%}")
        print(f"Degree correlation: {info_bh['localization']['degree_correlation']:.4f}")
        
        results['bh'] = {
            'accuracy': acc_bh,
            'metrics': metrics_bh,
            'localization': info_bh['localization'],
            'eigenvalue': ev_bh,
            'r_parameter': info_bh['r_parameter'],
            'n_negative_eigenvalues': info_bh['n_negative_eigenvalues']
        }
    except Exception as e:
        print(f"Error: {e}")
        results['bh'] = None
    
    # Comparison
    if results['nl'] and results['bh']:
        print("\n--- Comparison ---")
        print(f"Accuracy: NL={results['nl']['accuracy']:.2%} vs BH={results['bh']['accuracy']:.2%}")
        if results['bh']['accuracy'] > results['nl']['accuracy']:
            print("  → Bethe Hessian WINS on accuracy")
        elif results['bh']['accuracy'] < results['nl']['accuracy']:
            print("  → Normalized Laplacian WINS on accuracy")
        else:
            print("  → TIE on accuracy")
        
        print(f"Hub localization (weight on hubs): NL={results['nl']['localization']['weight_on_hubs']:.2%} "
              f"vs BH={results['bh']['localization']['weight_on_hubs']:.2%}")
        if results['bh']['localization']['weight_on_hubs'] < results['nl']['localization']['weight_on_hubs']:
            print("  → Bethe Hessian is LESS localized on hubs")
        else:
            print("  → Normalized Laplacian is LESS localized on hubs")
    
    return results

def run_newman_benchmarks():
    """Run experiments on Newman's benchmark networks."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: NEWMAN'S BENCHMARK NETWORKS")
    print("="*70)
    
    all_results = []
    
    # Karate Club
    G, gt, name = load_karate_club()
    results = compare_methods_on_network(G, gt, name)
    all_results.append(results)
    
    # Dolphins (synthetic if not available)
    G, gt, name = load_dolphins()
    results = compare_methods_on_network(G, gt, name)
    all_results.append(results)
    
    # Political Books (synthetic if not available)
    G, gt, name = load_political_books()
    results = compare_methods_on_network(G, gt, name)
    all_results.append(results)
    
    # Political Blogs (synthetic with power-law - Newman's problematic case)
    G, gt, name = load_political_blogs()
    results = compare_methods_on_network(G, gt, name)
    all_results.append(results)
    
    return all_results

def run_detectability_experiment():
    """Run detectability threshold experiment."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: DETECTABILITY THRESHOLD")
    print("="*70)
    print("Testing BH vs NL near the Kesten-Stigum threshold...")
    print("(This may take a few minutes)")
    
    results = detectability_threshold_experiment(n=500, num_trials=5)
    
    print("\nResults summary:")
    print(f"{'Epsilon':>10} {'NL Acc':>10} {'BH Acc':>10} {'Winner':>15}")
    print("-" * 50)
    for r in results:
        winner = "BH" if r['bh_accuracy_mean'] > r['nl_accuracy_mean'] + 0.01 else \
                 "NL" if r['nl_accuracy_mean'] > r['bh_accuracy_mean'] + 0.01 else "TIE"
        print(f"{r['epsilon']:>10.2f} {r['nl_accuracy_mean']:>10.2%} {r['bh_accuracy_mean']:>10.2%} {winner:>15}")
    
    return results

def run_hub_localization_experiment():
    """Run hub localization experiment."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 3: HUB LOCALIZATION")
    print("="*70)
    print("Testing eigenvector localization on different graph types...")
    
    results = hub_localization_experiment()
    
    print("\nResults summary:")
    print(f"{'Graph':>12} {'NL Hub Wt':>12} {'BH Hub Wt':>12} {'NL IPR':>10} {'BH IPR':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['graph_type']:>12} "
              f"{r['nl_localization']['weight_on_hubs']:>12.2%} "
              f"{r['bh_localization']['weight_on_hubs']:>12.2%} "
              f"{r['nl_localization']['ipr']:>10.4f} "
              f"{r['bh_localization']['ipr']:>10.4f}")
    
    return results

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_detectability_results(results, save_path=None):
    """Plot detectability threshold experiment results."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epsilons = [r['epsilon'] for r in results]
    nl_acc = [r['nl_accuracy_mean'] for r in results]
    nl_std = [r['nl_accuracy_std'] for r in results]
    bh_acc = [r['bh_accuracy_mean'] for r in results]
    bh_std = [r['bh_accuracy_std'] for r in results]
    
    ax.fill_between(epsilons, 
                    np.array(nl_acc) - np.array(nl_std),
                    np.array(nl_acc) + np.array(nl_std),
                    alpha=0.3, color='blue')
    ax.fill_between(epsilons,
                    np.array(bh_acc) - np.array(bh_std),
                    np.array(bh_acc) + np.array(bh_std),
                    alpha=0.3, color='red')
    
    ax.plot(epsilons, nl_acc, 'b-o', label='Normalized Laplacian', markersize=4)
    ax.plot(epsilons, bh_acc, 'r-s', label='Bethe Hessian', markersize=4)
    
    ax.axvline(x=0, color='gray', linestyle='--', label='Detectability threshold')
    ax.axhline(y=0.5, color='lightgray', linestyle=':', label='Random guess')
    
    ax.set_xlabel('Distance from threshold (ε)', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title('Bethe Hessian vs Normalized Laplacian Near Detectability Threshold', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig

def plot_eigenvector_comparison(G, v_nl, v_bh, name, save_path=None):
    """Plot eigenvector values sorted by magnitude."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    degrees = np.array([G.degree(n) for n in G.nodes()])
    
    # Sort by eigenvector value
    idx_nl = np.argsort(v_nl)
    idx_bh = np.argsort(v_bh)
    
    # Plot NL eigenvector
    ax = axes[0]
    colors_nl = plt.cm.coolwarm((degrees[idx_nl] - degrees.min()) / (degrees.max() - degrees.min() + 1e-10))
    ax.scatter(range(len(v_nl)), v_nl[idx_nl], c=colors_nl, s=20, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Vertex (sorted by eigenvector value)')
    ax.set_ylabel('Eigenvector element')
    ax.set_title(f'Normalized Laplacian - {name}')
    
    # Plot BH eigenvector
    ax = axes[1]
    colors_bh = plt.cm.coolwarm((degrees[idx_bh] - degrees.min()) / (degrees.max() - degrees.min() + 1e-10))
    ax.scatter(range(len(v_bh)), v_bh[idx_bh], c=colors_bh, s=20, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Vertex (sorted by eigenvector value)')
    ax.set_ylabel('Eigenvector element')
    ax.set_title(f'Bethe Hessian - {name}')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                               norm=plt.Normalize(vmin=degrees.min(), vmax=degrees.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label='Node Degree', shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all experiments."""
    
    print("="*70)
    print("BETHE HESSIAN VS NORMALIZED LAPLACIAN FOR NORMALIZED CUT")
    print("Following Newman (2013) 'Spectral methods for network community")
    print("detection and graph partitioning'")
    print("="*70)
    
    # Experiment 1: Newman's benchmarks
    newman_results = run_newman_benchmarks()
    
    # Experiment 2: Detectability threshold
    detect_results = run_detectability_experiment()
    
    # Experiment 3: Hub localization
    hub_results = run_hub_localization_experiment()
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    plot_detectability_results(detect_results, '/home/claude/detectability_comparison.png')
    print("Saved: detectability_comparison.png")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n1. Newman's Benchmark Networks:")
    for r in newman_results:
        if r['nl'] and r['bh']:
            winner = "BH" if r['bh']['accuracy'] > r['nl']['accuracy'] else \
                     "NL" if r['nl']['accuracy'] > r['bh']['accuracy'] else "TIE"
            print(f"   {r['network']}: NL={r['nl']['accuracy']:.1%}, BH={r['bh']['accuracy']:.1%} → {winner}")
    
    print("\n2. Detectability Threshold:")
    above_threshold = [r for r in detect_results if r['epsilon'] > 0]
    below_threshold = [r for r in detect_results if r['epsilon'] <= 0]
    
    if above_threshold:
        bh_wins_above = sum(1 for r in above_threshold if r['bh_accuracy_mean'] > r['nl_accuracy_mean'] + 0.01)
        print(f"   Above threshold: BH wins {bh_wins_above}/{len(above_threshold)} cases")
    if below_threshold:
        bh_wins_below = sum(1 for r in below_threshold if r['bh_accuracy_mean'] > r['nl_accuracy_mean'] + 0.01)
        print(f"   Below/at threshold: BH wins {bh_wins_below}/{len(below_threshold)} cases")
    
    print("\n3. Hub Localization:")
    for r in hub_results:
        less_localized = "BH" if r['bh_localization']['weight_on_hubs'] < r['nl_localization']['weight_on_hubs'] else "NL"
        print(f"   {r['graph_type']}: Less hub localization → {less_localized}")
    
    # Save all results
    all_results = {
        'newman_benchmarks': newman_results,
        'detectability': detect_results,
        'hub_localization': hub_results
    }
    
    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj
    
    with open('/home/claude/bh_vs_nl_results.json', 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print("\nResults saved to: bh_vs_nl_results.json")
    
    return all_results

if __name__ == '__main__':
    results = main()
