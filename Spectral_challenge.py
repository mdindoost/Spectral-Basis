#!/usr/bin/env python3
"""
Spectral Methods for Global Min-Cut: Starter Code
==================================================

This code provides implementations for experimenting with spectral
approaches to the global minimum cut problem.

Usage:
    python spectral_mincut_experiments.py

Requirements:
    pip install numpy scipy networkx matplotlib
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import networkx as nx
import time
from typing import Tuple, Set, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Core Algorithms
# ============================================================================

def sweep_cut(G: nx.Graph, vector: np.ndarray) -> Tuple[float, Set]:
    """
    Perform sweep cut on a vector to find minimum cut.
    
    Args:
        G: NetworkX graph
        vector: Node embedding vector (e.g., Fiedler vector)
    
    Returns:
        (cut_value, partition_S)
    """
    nodes = list(G.nodes())
    sorted_indices = np.argsort(vector)
    
    best_cut = float('inf')
    best_S = None
    
    # Try each threshold
    S = set()
    cut_edges = 0
    
    # Precompute degrees for efficiency
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    for i in range(len(nodes) - 1):
        node = nodes[sorted_indices[i]]
        S.add(node)
        
        # Update cut incrementally
        for neighbor in G.neighbors(node):
            if neighbor in S:
                cut_edges -= G[node][neighbor].get('weight', 1)
            else:
                cut_edges += G[node][neighbor].get('weight', 1)
        
        if cut_edges < best_cut:
            best_cut = cut_edges
            best_S = S.copy()
    
    return best_cut, best_S


def spectral_mincut_fiedler(G: nx.Graph) -> Tuple[float, Set, dict]:
    """
    Approximate min-cut using Fiedler vector (standard spectral method).
    
    Returns:
        (cut_value, partition_S, info_dict)
    """
    start = time.time()
    
    # Compute normalized Laplacian
    L = nx.normalized_laplacian_matrix(G).astype(float)
    
    # Get Fiedler vector (2nd smallest eigenvector)
    eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')
    fiedler = eigenvectors[:, 1]
    lambda2 = eigenvalues[1]
    
    eigen_time = time.time() - start
    
    # Sweep cut
    cut_value, partition = sweep_cut(G, fiedler)
    
    total_time = time.time() - start
    
    info = {
        'lambda2': lambda2,
        'eigen_time': eigen_time,
        'total_time': total_time,
        'method': 'fiedler'
    }
    
    return cut_value, partition, info


def spectral_mincut_bethe_hessian(G: nx.Graph, r: Optional[float] = None) -> Tuple[float, Set, dict]:
    """
    Approximate min-cut using Bethe Hessian matrix.
    
    The Bethe Hessian: H(r) = (r²-1)I - rA + D
    
    Args:
        G: NetworkX graph
        r: Bethe Hessian parameter (default: sqrt of average degree)
    
    Returns:
        (cut_value, partition_S, info_dict)
    """
    start = time.time()
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    # Adjacency and degree matrices
    A = nx.adjacency_matrix(G).astype(float)
    degrees = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    
    # Default r = sqrt(average degree)
    if r is None:
        avg_degree = 2 * m / n
        r = np.sqrt(avg_degree)
    
    # Bethe Hessian: H(r) = (r²-1)I - rA + D
    H = (r**2 - 1) * sp.eye(n) - r * A + D
    
    # Find smallest eigenvalues (may be negative)
    eigenvalues, eigenvectors = eigsh(H, k=3, which='SA')
    
    eigen_time = time.time() - start
    
    # Find eigenvector for most informative eigenvalue
    # (typically the one closest to but below 0, or the second smallest)
    idx = 1 if eigenvalues[0] < -0.01 else 0
    v = eigenvectors[:, idx]
    
    # Sweep cut
    cut_value, partition = sweep_cut(G, v)
    
    total_time = time.time() - start
    
    info = {
        'r': r,
        'eigenvalues': eigenvalues[:3],
        'eigen_time': eigen_time,
        'total_time': total_time,
        'method': 'bethe_hessian'
    }
    
    return cut_value, partition, info


def spectral_mincut_multi_eigenvector(G: nx.Graph, k: int = 3) -> Tuple[float, Set, dict]:
    """
    Min-cut using multiple eigenvectors (higher-order spectral method).
    
    Args:
        G: NetworkX graph
        k: Number of eigenvectors to use
    
    Returns:
        (cut_value, partition_S, info_dict)
    """
    start = time.time()
    
    L = nx.normalized_laplacian_matrix(G).astype(float)
    eigenvalues, eigenvectors = eigsh(L, k=k+1, which='SM')
    
    eigen_time = time.time() - start
    
    # Embedding: each node as point in R^k
    # Skip first eigenvector (constant)
    embedding = eigenvectors[:, 1:k+1]
    
    # Normalize rows (as in Ng et al. spectral clustering)
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    embedding = embedding / row_norms
    
    # Try sweep cuts in each dimension
    best_cut = float('inf')
    best_partition = None
    
    for dim in range(k):
        cut, partition = sweep_cut(G, embedding[:, dim])
        if cut < best_cut:
            best_cut = cut
            best_partition = partition
    
    total_time = time.time() - start
    
    info = {
        'k': k,
        'eigenvalues': eigenvalues[:k+1],
        'spectral_gap': eigenvalues[k] - eigenvalues[1] if k > 1 else 0,
        'eigen_time': eigen_time,
        'total_time': total_time,
        'method': f'multi_eigenvector_k{k}'
    }
    
    return best_cut, best_partition, info


def exact_mincut_stoer_wagner(G: nx.Graph) -> Tuple[float, Set, dict]:
    """
    Exact global min-cut using Stoer-Wagner algorithm (via NetworkX).
    
    Returns:
        (cut_value, partition_S, info_dict)
    """
    start = time.time()
    
    cut_value, partition = nx.stoer_wagner(G)
    
    total_time = time.time() - start
    
    info = {
        'total_time': total_time,
        'method': 'stoer_wagner'
    }
    
    return cut_value, set(partition[0]), info


# ============================================================================
# Hybrid Methods
# ============================================================================

def hybrid_spectral_exact(G: nx.Graph, boundary_hops: int = 2) -> Tuple[float, Set, dict]:
    """
    Hybrid approach: spectral initialization + local exact refinement.
    
    1. Get approximate cut from Fiedler vector
    2. Identify boundary region
    3. Refine locally
    
    Args:
        G: NetworkX graph
        boundary_hops: Number of hops to include in boundary region
    
    Returns:
        (cut_value, partition_S, info_dict)
    """
    start = time.time()
    
    # Step 1: Spectral approximation
    spectral_cut, spectral_S, spectral_info = spectral_mincut_fiedler(G)
    
    # Step 2: Identify boundary
    boundary = set()
    for node in spectral_S:
        for neighbor in G.neighbors(node):
            if neighbor not in spectral_S:
                boundary.add(node)
                boundary.add(neighbor)
    
    # Expand boundary
    for _ in range(boundary_hops - 1):
        new_boundary = set()
        for node in boundary:
            for neighbor in G.neighbors(node):
                new_boundary.add(neighbor)
        boundary.update(new_boundary)
    
    # Step 3: If boundary is small enough, refine with exact method
    if len(boundary) < 0.5 * G.number_of_nodes():
        # Extract subgraph
        subgraph = G.subgraph(boundary).copy()
        
        if subgraph.number_of_nodes() > 2 and nx.is_connected(subgraph):
            try:
                local_cut, local_partition = nx.stoer_wagner(subgraph)
                
                # Merge local solution with global spectral solution
                local_S = set(local_partition[0])
                
                # Decide which side of local cut aligns with spectral_S
                overlap_0 = len(local_S & spectral_S)
                overlap_1 = len(set(local_partition[1]) & spectral_S)
                
                if overlap_0 >= overlap_1:
                    refined_S = (spectral_S - boundary) | local_S
                else:
                    refined_S = (spectral_S - boundary) | set(local_partition[1])
                
                # Compute actual cut value
                refined_cut = nx.cut_size(G, refined_S)
                
                if refined_cut < spectral_cut:
                    total_time = time.time() - start
                    return refined_cut, refined_S, {
                        'spectral_cut': spectral_cut,
                        'refined_cut': refined_cut,
                        'boundary_size': len(boundary),
                        'total_time': total_time,
                        'method': 'hybrid'
                    }
            except:
                pass
    
    total_time = time.time() - start
    return spectral_cut, spectral_S, {
        'spectral_cut': spectral_cut,
        'boundary_size': len(boundary),
        'total_time': total_time,
        'method': 'hybrid (no refinement)'
    }


# ============================================================================
# Experimental Framework
# ============================================================================

def run_experiments(G: nx.Graph, name: str = "Graph"):
    """
    Run all min-cut methods on a graph and compare results.
    """
    print(f"\n{'='*60}")
    print(f"Graph: {name}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"{'='*60}")
    
    results = {}
    
    # 1. Exact (Stoer-Wagner)
    try:
        cut, S, info = exact_mincut_stoer_wagner(G)
        results['stoer_wagner'] = {'cut': cut, 'time': info['total_time']}
        print(f"\nStoer-Wagner (exact):")
        print(f"  Cut value: {cut}")
        print(f"  Time: {info['total_time']:.4f}s")
        exact_cut = cut
    except Exception as e:
        print(f"\nStoer-Wagner failed: {e}")
        exact_cut = None
    
    # 2. Fiedler (standard spectral)
    try:
        cut, S, info = spectral_mincut_fiedler(G)
        results['fiedler'] = {'cut': cut, 'time': info['total_time'], 'lambda2': info['lambda2']}
        ratio = cut / exact_cut if exact_cut else "N/A"
        print(f"\nFiedler (spectral):")
        print(f"  Cut value: {cut}")
        print(f"  λ₂: {info['lambda2']:.6f}")
        print(f"  Approx ratio: {ratio}")
        print(f"  Time: {info['total_time']:.4f}s")
    except Exception as e:
        print(f"\nFiedler failed: {e}")
    
    # 3. Bethe Hessian
    try:
        cut, S, info = spectral_mincut_bethe_hessian(G)
        results['bethe_hessian'] = {'cut': cut, 'time': info['total_time'], 'r': info['r']}
        ratio = cut / exact_cut if exact_cut else "N/A"
        print(f"\nBethe Hessian:")
        print(f"  Cut value: {cut}")
        print(f"  r parameter: {info['r']:.4f}")
        print(f"  Approx ratio: {ratio}")
        print(f"  Time: {info['total_time']:.4f}s")
    except Exception as e:
        print(f"\nBethe Hessian failed: {e}")
    
    # 4. Multi-eigenvector
    try:
        cut, S, info = spectral_mincut_multi_eigenvector(G, k=3)
        results['multi_ev'] = {'cut': cut, 'time': info['total_time']}
        ratio = cut / exact_cut if exact_cut else "N/A"
        print(f"\nMulti-eigenvector (k=3):")
        print(f"  Cut value: {cut}")
        print(f"  Spectral gap: {info['spectral_gap']:.6f}")
        print(f"  Approx ratio: {ratio}")
        print(f"  Time: {info['total_time']:.4f}s")
    except Exception as e:
        print(f"\nMulti-eigenvector failed: {e}")
    
    # 5. Hybrid
    try:
        cut, S, info = hybrid_spectral_exact(G)
        results['hybrid'] = {'cut': cut, 'time': info['total_time']}
        ratio = cut / exact_cut if exact_cut else "N/A"
        print(f"\nHybrid (spectral + local exact):")
        print(f"  Cut value: {cut}")
        print(f"  Boundary size: {info.get('boundary_size', 'N/A')}")
        print(f"  Approx ratio: {ratio}")
        print(f"  Time: {info['total_time']:.4f}s")
    except Exception as e:
        print(f"\nHybrid failed: {e}")
    
    return results


def compute_graph_properties(G: nx.Graph) -> dict:
    """Compute graph properties relevant to spectral min-cut."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    # Degree statistics
    degrees = [d for _, d in G.degree()]
    avg_degree = 2 * m / n
    max_degree = max(degrees)
    degree_variance = np.var(degrees)
    
    # Spectral properties
    L = nx.normalized_laplacian_matrix(G).astype(float)
    eigenvalues = eigsh(L, k=min(5, n-1), which='SM', return_eigenvectors=False)
    eigenvalues = np.sort(eigenvalues)
    
    return {
        'n': n,
        'm': m,
        'density': 2 * m / (n * (n-1)),
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'degree_variance': degree_variance,
        'lambda2': eigenvalues[1] if len(eigenvalues) > 1 else None,
        'lambda3': eigenvalues[2] if len(eigenvalues) > 2 else None,
        'spectral_gap_2_3': eigenvalues[2] - eigenvalues[1] if len(eigenvalues) > 2 else None
    }


# ============================================================================
# Test Graphs
# ============================================================================

def generate_test_graphs():
    """Generate various test graphs for experiments."""
    graphs = {}
    
    # 1. Barbell graph (clear min-cut)
    graphs['barbell_20'] = nx.barbell_graph(10, 0)
    
    # 2. Random regular graph
    graphs['random_regular_100'] = nx.random_regular_graph(4, 100, seed=42)
    
    # 3. Erdos-Renyi
    G = nx.erdos_renyi_graph(100, 0.1, seed=42)
    if nx.is_connected(G):
        graphs['erdos_renyi_100'] = G
    else:
        graphs['erdos_renyi_100'] = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    
    # 4. Stochastic block model (2 communities)
    sizes = [50, 50]
    p_in, p_out = 0.3, 0.05
    G = nx.stochastic_block_model(sizes, [[p_in, p_out], [p_out, p_in]], seed=42)
    graphs['sbm_2_communities'] = G
    
    # 5. Karate club (classic)
    graphs['karate_club'] = nx.karate_club_graph()
    
    # 6. Power-law cluster graph
    G = nx.powerlaw_cluster_graph(100, 3, 0.5, seed=42)
    if nx.is_connected(G):
        graphs['powerlaw_100'] = G
    
    return graphs


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Spectral Methods for Global Min-Cut: Experiments")
    print("=" * 60)
    
    # Generate test graphs
    graphs = generate_test_graphs()
    
    all_results = {}
    
    for name, G in graphs.items():
        if not nx.is_connected(G):
            print(f"\nSkipping {name}: not connected")
            continue
        
        # Compute properties
        props = compute_graph_properties(G)
        print(f"\n{name} properties:")
        print(f"  λ₂ = {props['lambda2']:.6f}")
        print(f"  Spectral gap (λ₃-λ₂) = {props['spectral_gap_2_3']:.6f}")
        print(f"  Degree variance = {props['degree_variance']:.2f}")
        
        # Run experiments
        results = run_experiments(G, name)
        all_results[name] = {'properties': props, 'results': results}
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, data in all_results.items():
        print(f"\n{name}:")
        results = data['results']
        exact = results.get('stoer_wagner', {}).get('cut')
        if exact:
            for method, r in results.items():
                if method != 'stoer_wagner':
                    ratio = r['cut'] / exact
                    speedup = results['stoer_wagner']['time'] / r['time']
                    print(f"  {method}: ratio={ratio:.3f}, speedup={speedup:.2f}x")
