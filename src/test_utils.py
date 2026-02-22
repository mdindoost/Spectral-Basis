"""
Tests for src/utils.py

Run with:
    cd /home/md724/Spectral-Basis
    venv/bin/python -m pytest src/test_utils.py -v

Priority: build_graph_matrices and compute_restricted_eigenvectors are the
most critical (the matrix-swap bug lived in their interaction).
"""

import sys
import os
import warnings

import pytest
import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    build_graph_matrices,
    get_largest_connected_component_nx,
    extract_subgraph,
    compute_sgc_normalized_adjacency,
    sgc_precompute,
    compute_restricted_eigenvectors,
    StandardMLP,
    RowNormMLP,
    LogMagnitudeMLP,
    SpectralRowNormMLP,
    NestedSpheresClassifier,
    NestedSpheresMLP,
)


# ============================================================================
# Shared graph helpers
# ============================================================================

def triangle_edge_index():
    """3-node fully-connected triangle: edges 0-1, 1-2, 0-2 (undirected pairs)"""
    return np.array([[0, 1, 1, 2, 0, 2],
                     [1, 0, 2, 1, 2, 0]])


def path_edge_index(n=5):
    """n-node path: 0—1—2—...—(n-1)"""
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    return np.array([src, dst])


def two_component_adj():
    """
    Disconnected graph: component A = {0,1,2} (triangle),
                        component B = {3,4} (single edge).
    No self-loops (raw adjacency for LCC tests).
    """
    rows = [0, 1, 1, 2, 0, 2,  3, 4]
    cols = [1, 0, 2, 1, 2, 0,  4, 3]
    data = np.ones(len(rows))
    return sp.coo_matrix((data, (rows, cols)), shape=(5, 5)).tocsr()


# ============================================================================
# 1. build_graph_matrices
# ============================================================================

class TestBuildGraphMatrices:

    def test_return_order_L_has_negative_offdiag_D_is_diagonal(self):
        """Return order is (adj, L, D): L has negative off-diagonals, D is diagonal-only."""
        adj, L, D = build_graph_matrices(triangle_edge_index(), 3)

        L_dense = L.toarray()
        D_dense = D.toarray()

        off_L = L_dense - np.diag(np.diag(L_dense))
        assert np.all(off_L <= 1e-12), "L off-diagonals must be <= 0 (Laplacian)"

        off_D = D_dense - np.diag(np.diag(D_dense))
        assert np.allclose(off_D, 0), "D must be diagonal-only"

    def test_adjacency_symmetric(self):
        adj, L, D = build_graph_matrices(path_edge_index(4), 4)
        assert np.allclose((adj - adj.T).toarray(), 0), "adj must be symmetric"

    def test_self_loops_added(self):
        adj, L, D = build_graph_matrices(triangle_edge_index(), 3)
        assert np.all(adj.diagonal() == 1), "All diagonal entries of adj must be 1"

    def test_degree_matrix_matches_row_sums(self):
        adj, L, D = build_graph_matrices(path_edge_index(5), 5)
        row_sums = np.array(adj.sum(axis=1)).ravel()
        assert np.allclose(D.diagonal(), row_sums), "D diagonal must equal adj row sums"

    def test_laplacian_equals_D_minus_adj(self):
        adj, L, D = build_graph_matrices(triangle_edge_index(), 3)
        expected = (D - adj).toarray()
        assert np.allclose(L.toarray(), expected), "L must equal D - adj"

    def test_laplacian_is_PSD(self):
        """All eigenvalues of L must be >= 0."""
        adj, L, D = build_graph_matrices(path_edge_index(6), 6)
        eigs = np.linalg.eigvalsh(L.toarray())
        assert np.all(eigs >= -1e-10), f"L must be PSD, min eigenvalue = {eigs.min():.2e}"

    def test_laplacian_row_sums_zero(self):
        """L @ 1 must be 0 (standard Laplacian property)."""
        adj, L, D = build_graph_matrices(triangle_edge_index(), 3)
        result = L @ np.ones(3)
        assert np.allclose(result, 0, atol=1e-10), "L @ 1 must be zero vector"

    def test_triangle_known_degrees(self):
        """Triangle + self-loops: each node touches 2 neighbours + itself → degree 3."""
        adj, L, D = build_graph_matrices(triangle_edge_index(), 3)
        assert np.allclose(D.diagonal(), [3, 3, 3])

    def test_path_3_known_degrees(self):
        """Path 0—1—2 with self-loops: degrees [2, 3, 2]."""
        adj, L, D = build_graph_matrices(path_edge_index(3), 3)
        assert np.allclose(D.diagonal(), [2, 3, 2])

    def test_isolated_node_degree_one(self):
        """Single isolated node (no edges) has degree 1 after self-loop."""
        adj, L, D = build_graph_matrices(np.zeros((2, 0), dtype=int), 1)
        assert D.diagonal()[0] == 1


# ============================================================================
# 2. compute_restricted_eigenvectors
# ============================================================================

@pytest.fixture(scope="module")
def path5_LD():
    """L and D for a 5-node path graph (reused across tests)."""
    _, L, D = build_graph_matrices(path_edge_index(5), 5)
    return L, D


class TestComputeRestrictedEigenvectors:

    def test_d_orthonormality(self, path5_LD):
        """U^T D U must be identity (D-orthonormal eigenvectors)."""
        L, D = path5_LD
        np.random.seed(0)
        X = np.random.randn(5, 3)
        U, eigs, d_eff, ortho_err = compute_restricted_eigenvectors(X, L, D)
        G = U.T @ D.toarray() @ U
        assert np.allclose(G, np.eye(d_eff), atol=1e-6), \
            f"D-orthonormality failed, max error = {ortho_err:.2e}"

    def test_eigenvalue_range_in_0_2(self, path5_LD):
        """All eigenvalues must lie in [0, 2] for normalized Laplacian."""
        L, D = path5_LD
        np.random.seed(1)
        X = np.random.randn(5, 3)
        U, eigs, d_eff, ortho_err = compute_restricted_eigenvectors(X, L, D)
        assert eigs.min() >= -0.01, f"Min eigenvalue {eigs.min():.4f} below 0"
        assert eigs.max() <= 2.1,   f"Max eigenvalue {eigs.max():.4f} above 2"

    def test_rayleigh_quotient(self, path5_LD):
        """u_i^T L u_i / u_i^T D u_i must equal lambda_i for each eigenvector."""
        L, D = path5_LD
        np.random.seed(2)
        X = np.random.randn(5, 4)
        U, eigs, d_eff, ortho_err = compute_restricted_eigenvectors(X, L, D)
        L_d = L.toarray()
        D_d = D.toarray()
        for i in range(d_eff):
            u = U[:, i]
            rq = (u @ L_d @ u) / (u @ D_d @ u)
            assert abs(rq - eigs[i]) < 1e-6, \
                f"Rayleigh quotient mismatch at eigenvector {i}: rq={rq:.6f}, λ={eigs[i]:.6f}"

    def test_ortho_error_small(self, path5_LD):
        """Reported ortho_error must be below 1e-6."""
        L, D = path5_LD
        np.random.seed(3)
        X = np.random.randn(5, 3)
        U, eigs, d_eff, ortho_err = compute_restricted_eigenvectors(X, L, D)
        assert ortho_err < 1e-6, f"ortho_error too large: {ortho_err:.2e}"

    def test_swap_alarm_fires(self, path5_LD):
        """Passing L and D swapped must trigger the EIGENVALUE ALARM warning."""
        L, D = path5_LD
        np.random.seed(4)
        X = np.random.randn(5, 3)
        with pytest.warns(UserWarning, match="out of expected"):
            compute_restricted_eigenvectors(X, D, L)  # intentionally swapped

    def test_no_alarm_on_correct_input(self, path5_LD):
        """Correct L and D must not trigger the eigenvalue alarm."""
        L, D = path5_LD
        np.random.seed(5)
        X = np.random.randn(5, 3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_restricted_eigenvectors(X, L, D)
        alarm_warnings = [w for w in caught if "ALARM" in str(w.message)]
        assert len(alarm_warnings) == 0, "No ALARM warning should fire for correct L, D"

    def test_rank_deficient_X_reduces_d_effective(self, path5_LD):
        """Rank-2 X with 4 columns → d_effective should be 2."""
        L, D = path5_LD
        np.random.seed(6)
        base = np.random.randn(5, 2)
        X = np.hstack([base, base])  # rank 2, 4 columns
        U, eigs, d_eff, ortho_err = compute_restricted_eigenvectors(X, L, D)
        assert d_eff == 2, f"Expected d_effective=2 for rank-2 X, got {d_eff}"

    def test_num_components_drops_smallest_eigenvalue(self, path5_LD):
        """num_components=1 should drop the smallest eigenvalue."""
        L, D = path5_LD
        np.random.seed(7)
        X = np.random.randn(5, 4)
        U0, eigs0, d0, _ = compute_restricted_eigenvectors(X, L, D, num_components=0)
        U1, eigs1, d1, _ = compute_restricted_eigenvectors(X, L, D, num_components=1)
        assert d1 == d0 - 1, "num_components=1 must reduce d_effective by 1"
        assert np.allclose(eigs1, eigs0[1:]), \
            "Remaining eigenvalues must equal eigs0[1:] (smallest dropped)"

    def test_subspace_constraint(self, path5_LD):
        """All columns of U must lie in the column space of X."""
        L, D = path5_LD
        np.random.seed(8)
        X = np.random.randn(5, 3)
        U, eigs, d_eff, ortho_err = compute_restricted_eigenvectors(X, L, D)
        Q, _ = np.linalg.qr(X)
        Q = Q[:, :np.linalg.matrix_rank(X)]
        U_projected = Q @ (Q.T @ U)
        assert np.allclose(U, U_projected, atol=1e-6), \
            "U columns must lie in column space of X"

    def test_d_effective_returned_correctly(self, path5_LD):
        """d_effective must equal the number of columns in U."""
        L, D = path5_LD
        np.random.seed(9)
        X = np.random.randn(5, 3)
        U, eigs, d_eff, ortho_err = compute_restricted_eigenvectors(X, L, D)
        assert U.shape[1] == d_eff, "d_effective must equal number of columns in U"
        assert len(eigs) == d_eff, "len(eigenvalues) must equal d_effective"


# ============================================================================
# 3. get_largest_connected_component_nx
# ============================================================================

class TestGetLargestConnectedComponent:

    def test_already_connected_returns_all_true(self):
        adj, L, D = build_graph_matrices(triangle_edge_index(), 3)
        mask = get_largest_connected_component_nx(adj)
        assert np.all(mask), "Connected graph → all-True mask"

    def test_two_components_selects_larger(self):
        """Component {0,1,2} (size 3) vs {3,4} (size 2) → mask selects first 3 nodes."""
        adj = two_component_adj()
        mask = get_largest_connected_component_nx(adj)
        assert mask.sum() == 3,         f"LCC should have 3 nodes, got {mask.sum()}"
        assert np.all(mask[:3]),        "Nodes 0,1,2 must be in LCC"
        assert not np.any(mask[3:]),    "Nodes 3,4 must not be in LCC"

    def test_all_isolated_returns_one_node(self):
        """4 completely isolated nodes: LCC has size 1."""
        adj = sp.eye(4, format='csr') * 0   # no edges, no self-loops
        # Add nothing — each node is truly isolated
        adj = sp.csr_matrix((4, 4))
        mask = get_largest_connected_component_nx(adj)
        assert mask.sum() == 1, f"All isolated → LCC size 1, got {mask.sum()}"

    def test_mask_dtype_bool(self):
        adj, L, D = build_graph_matrices(path_edge_index(4), 4)
        mask = get_largest_connected_component_nx(adj)
        assert mask.dtype == bool, "Mask must be boolean"


# ============================================================================
# 4. extract_subgraph
# ============================================================================

class TestExtractSubgraph:

    def test_shape_of_subgraph(self):
        adj, L, D = build_graph_matrices(triangle_edge_index(), 3)
        features = np.eye(3)
        labels = np.array([0, 1, 2])
        mask = np.array([True, True, False])
        adj_sub, feat_sub, lab_sub, _ = extract_subgraph(adj, features, labels, mask, None)
        assert adj_sub.shape == (2, 2)
        assert feat_sub.shape == (2, 3)
        assert lab_sub.shape == (2,)

    def test_features_and_labels_correct_rows(self):
        adj, L, D = build_graph_matrices(triangle_edge_index(), 3)
        features = np.arange(9).reshape(3, 3).astype(float)
        labels = np.array([10, 20, 30])
        mask = np.array([True, False, True])
        _, feat_sub, lab_sub, _ = extract_subgraph(adj, features, labels, mask, None)
        assert np.allclose(feat_sub, features[[0, 2]])
        assert np.array_equal(lab_sub, [10, 30])

    def test_split_idx_none_passthrough(self):
        adj, L, D = build_graph_matrices(triangle_edge_index(), 3)
        mask = np.array([True, True, False])
        _, _, _, split_sub = extract_subgraph(adj, np.eye(3), np.zeros(3), mask, None)
        assert split_sub is None

    def test_split_indices_remapped(self):
        """Old node 0→new 0, old node 2→new 1, old node 3→new 2."""
        adj, L, D = build_graph_matrices(path_edge_index(5), 5)
        mask = np.array([True, False, True, True, False])  # keeps nodes 0,2,3
        features = np.eye(5)
        labels = np.zeros(5, dtype=int)
        split_idx = {'train': np.array([0, 2]), 'test': np.array([3])}
        _, _, _, split_sub = extract_subgraph(adj, features, labels, mask, split_idx)
        assert set(split_sub['train']) == {0, 1}, \
            f"train should be {{0,1}}, got {set(split_sub['train'])}"
        assert set(split_sub['test']) == {2}, \
            f"test should be {{2}}, got {set(split_sub['test'])}"

    def test_nodes_not_in_mask_excluded_from_splits(self):
        """Nodes outside the mask must be dropped from all split arrays."""
        adj, L, D = build_graph_matrices(path_edge_index(5), 5)
        mask = np.array([True, False, True, True, False])
        features = np.eye(5)
        labels = np.zeros(5, dtype=int)
        # node 1 and 4 are outside the mask
        split_idx = {'train': np.array([0, 1, 2]), 'test': np.array([3, 4])}
        _, _, _, split_sub = extract_subgraph(adj, features, labels, mask, split_idx)
        assert len(split_sub['train']) == 2, "Node 1 (outside mask) should be dropped from train"
        assert len(split_sub['test']) == 1,  "Node 4 (outside mask) should be dropped from test"


# ============================================================================
# 5. compute_sgc_normalized_adjacency
# ============================================================================

class TestComputeSGCNormalizedAdjacency:

    def test_symmetric(self):
        adj, L, D = build_graph_matrices(path_edge_index(5), 5)
        A_sgc = compute_sgc_normalized_adjacency(adj)
        diff = (A_sgc - A_sgc.T).toarray()
        assert np.allclose(diff, 0, atol=1e-10), "SGC normalized adj must be symmetric"

    def test_eigenvalue_range_minus1_to_1(self):
        """Eigenvalues of symmetric normalized adjacency must be in [-1, 1]."""
        adj, L, D = build_graph_matrices(path_edge_index(6), 6)
        A_sgc = compute_sgc_normalized_adjacency(adj)
        eigs = np.linalg.eigvalsh(A_sgc.toarray())
        assert eigs.min() >= -1.01, f"Min eigenvalue {eigs.min():.4f} below -1"
        assert eigs.max() <=  1.01, f"Max eigenvalue {eigs.max():.4f} above +1"

    def test_output_shape(self):
        adj, L, D = build_graph_matrices(path_edge_index(4), 4)
        A_sgc = compute_sgc_normalized_adjacency(adj)
        assert A_sgc.shape == (4, 4)


# ============================================================================
# 6. sgc_precompute
# ============================================================================

class TestSGCPrecompute:

    @pytest.fixture(autouse=True)
    def setup(self):
        adj, L, D = build_graph_matrices(path_edge_index(5), 5)
        self.A = compute_sgc_normalized_adjacency(adj)
        np.random.seed(42)
        self.X = np.random.randn(5, 7)

    def test_degree_zero_unchanged(self):
        result = sgc_precompute(self.X.copy(), self.A, 0)
        assert np.allclose(result, self.X), "degree=0 must return features unchanged"

    def test_degree_one_equals_A_X(self):
        result = sgc_precompute(self.X.copy(), self.A, 1)
        expected = self.A @ self.X
        assert np.allclose(result, expected), "degree=1 must equal A @ X"

    def test_degree_two_equals_A2_X(self):
        result = sgc_precompute(self.X.copy(), self.A, 2)
        expected = self.A @ (self.A @ self.X)
        assert np.allclose(result, expected), "degree=2 must equal A^2 @ X"

    def test_shape_preserved(self):
        result = sgc_precompute(self.X.copy(), self.A, 3)
        assert result.shape == self.X.shape, "Output shape must match input shape"


# ============================================================================
# 7. Model architectures
# ============================================================================

N, D_IN, H, C = 20, 10, 32, 3
EIGS = torch.linspace(0.1, 1.9, D_IN)


class TestModelArchitectures:

    def _input(self):
        torch.manual_seed(0)
        return torch.randn(N, D_IN)

    def test_standard_mlp_output_shape(self):
        out = StandardMLP(D_IN, H, C)(self._input())
        assert out.shape == (N, C)

    def test_rownorm_mlp_output_shape(self):
        out = RowNormMLP(D_IN, H, C)(self._input())
        assert out.shape == (N, C)

    def test_rownorm_mlp_has_no_bias(self):
        model = RowNormMLP(D_IN, H, C)
        bias_params = [n for n, _ in model.named_parameters() if 'bias' in n]
        assert len(bias_params) == 0, f"RowNormMLP must have no bias, found: {bias_params}"

    def test_log_magnitude_mlp_output_shape(self):
        out = LogMagnitudeMLP(D_IN, H, C)(self._input())
        assert out.shape == (N, C)

    def test_spectral_rownorm_mlp_output_shape(self):
        out = SpectralRowNormMLP(D_IN, H, C, EIGS, alpha=0.5)(self._input())
        assert out.shape == (N, C)

    def test_spectral_rownorm_mlp_alpha_zero_gives_ones_weights(self):
        """alpha=0 must produce all-ones eigenvalue weights (no spectral bias)."""
        model = SpectralRowNormMLP(D_IN, H, C, EIGS, alpha=0.0)
        assert torch.allclose(model.eigenvalue_weights, torch.ones(D_IN)), \
            "alpha=0 eigenvalue weights must be all-ones"

    def test_nested_spheres_classifier_output_shape(self):
        out = NestedSpheresClassifier(D_IN, H, C, EIGS, alpha=0.5, beta=1.0)(self._input())
        assert out.shape == (N, C)

    def test_nested_spheres_mlp_output_shape(self):
        out = NestedSpheresMLP(D_IN, H, C, EIGS, alpha=0.5, beta=1.0)(self._input())
        assert out.shape == (N, C)

    def test_no_nan_or_inf_in_any_model(self):
        """All models must produce finite outputs on random input."""
        x = self._input()
        models = [
            StandardMLP(D_IN, H, C),
            RowNormMLP(D_IN, H, C),
            LogMagnitudeMLP(D_IN, H, C),
            SpectralRowNormMLP(D_IN, H, C, EIGS, alpha=0.5),
            NestedSpheresClassifier(D_IN, H, C, EIGS, alpha=0.5, beta=1.0),
            NestedSpheresMLP(D_IN, H, C, EIGS, alpha=0.5, beta=1.0),
        ]
        for model in models:
            out = model(x)
            assert torch.isfinite(out).all(), \
                f"{model.__class__.__name__} produced NaN or Inf in output"

    def test_sgc_output_shape(self):
        from utils import SGC
        out = SGC(D_IN, C)(self._input())
        assert out.shape == (N, C)
