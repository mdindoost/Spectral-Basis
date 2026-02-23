"""
Tests for src/graph_utils.py and src/models.py

Run with:
    cd /home/md724/Spectral-Basis
    venv/bin/python -m pytest src/test_utils.py -v

Priority: build_graph_matrices and compute_restricted_eigenvectors are the
most critical (the matrix-swap bug lived in their interaction).
"""

import sys
import os
import copy
import warnings

import pytest
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from graph_utils import (
    load_dataset,
    build_graph_matrices,
    get_largest_connected_component_nx,
    extract_subgraph,
    compute_sgc_normalized_adjacency,
    sgc_precompute,
    compute_restricted_eigenvectors,
)

from models import (
    SGC,
    StandardMLP,
    RowNormMLP,
    CosineRowNormMLP,
    LogMagnitudeMLP,
    DualStreamMLP,
    SpectralRowNormMLP,
    NestedSpheresMLP,
    NestedSpheresClassifier,
    train_simple,
    train_with_selection,
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
# 1. load_dataset
# ============================================================================

class TestLoadDataset:

    def test_unknown_dataset_raises_value_error(self):
        """Requesting a non-existent dataset must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("not_a_real_dataset_xyz")

    def test_unknown_dataset_case_insensitive_check(self):
        """Dataset name is lowercased internally — unknown names must still raise."""
        with pytest.raises(ValueError):
            load_dataset("TOTALLY_FAKE_123")


# ============================================================================
# 2. build_graph_matrices
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
# 3. compute_restricted_eigenvectors
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
# 4. get_largest_connected_component_nx
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
        adj = sp.csr_matrix((4, 4))
        mask = get_largest_connected_component_nx(adj)
        assert mask.sum() == 1, f"All isolated → LCC size 1, got {mask.sum()}"

    def test_mask_dtype_bool(self):
        adj, L, D = build_graph_matrices(path_edge_index(4), 4)
        mask = get_largest_connected_component_nx(adj)
        assert mask.dtype == bool, "Mask must be boolean"


# ============================================================================
# 5. extract_subgraph
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

    def test_lcc_rebuild_no_double_self_loop(self):
        """Full LCC rebuild pipeline must produce adj with diagonal exactly 1.

        CRIT-1 bug: extract_subgraph preserves self-loops from the first
        build_graph_matrices call. If adj.tocoo() is passed directly back into
        build_graph_matrices, the (i,i) entries appear in the edge list and
        sp.eye is added a second time — diagonal becomes 2 instead of 1.
        The correct pipeline strips self-loops before rebuilding.
        """
        adj, L, D = build_graph_matrices(path_edge_index(5), 5)
        mask = np.ones(5, dtype=bool)  # keep all nodes (trivial LCC)
        adj_sub, _, _, _ = extract_subgraph(adj, np.eye(5), np.zeros(5), mask, None)

        # Correct: strip self-loops before rebuild
        adj_no_loops = adj_sub - sp.diags(adj_sub.diagonal())
        adj_coo = adj_no_loops.tocoo()
        edge_index_lcc = np.vstack([adj_coo.row, adj_coo.col])
        adj_rebuilt, _, _ = build_graph_matrices(edge_index_lcc, adj_sub.shape[0])

        assert np.allclose(adj_rebuilt.diagonal(), 1.0), (
            f"LCC rebuild diagonal must be 1, got {adj_rebuilt.diagonal()} "
            f"(double self-loop bug: CRIT-1)"
        )


# ============================================================================
# 6. compute_sgc_normalized_adjacency
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

    def test_diagonal_equals_1_over_deg_plus_1(self):
        """Diagonal of A_sgc must be 1/(deg+1) — SGC paper formula, no double self-loop."""
        adj, L, D = build_graph_matrices(path_edge_index(5), 5)
        A_sgc = compute_sgc_normalized_adjacency(adj)
        true_deg = np.array(adj.sum(axis=1)).ravel() - 1.0  # subtract self-loop
        expected = 1.0 / (true_deg + 1.0)
        assert np.allclose(A_sgc.diagonal(), expected, atol=1e-10), \
            "A_sgc diagonal must be 1/(deg+1), not 2/(deg+2) (double self-loop bug)"


# ============================================================================
# 7. sgc_precompute
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
# 8. Model architectures
# ============================================================================

N, D_IN, H, C = 20, 10, 32, 3
H_MAG = 8
EIGS = torch.linspace(0.1, 1.9, D_IN)


class TestModelArchitectures:

    def _input(self):
        torch.manual_seed(0)
        return torch.randn(N, D_IN)

    # ── SGC ──────────────────────────────────────────────────────────────────

    def test_sgc_output_shape(self):
        out = SGC(D_IN, C)(self._input())
        assert out.shape == (N, C)

    def test_sgc_has_bias(self):
        model = SGC(D_IN, C)
        assert model.W.bias is not None, "SGC must have bias"

    # ── StandardMLP ──────────────────────────────────────────────────────────

    def test_standard_mlp_output_shape(self):
        out = StandardMLP(D_IN, H, C)(self._input())
        assert out.shape == (N, C)

    # ── RowNormMLP ───────────────────────────────────────────────────────────

    def test_rownorm_mlp_output_shape(self):
        out = RowNormMLP(D_IN, H, C)(self._input())
        assert out.shape == (N, C)

    def test_rownorm_mlp_has_no_bias(self):
        model = RowNormMLP(D_IN, H, C)
        bias_params = [n for n, _ in model.named_parameters() if 'bias' in n]
        assert len(bias_params) == 0, f"RowNormMLP must have no bias, found: {bias_params}"

    # ── CosineRowNormMLP ─────────────────────────────────────────────────────

    def test_cosine_rownorm_mlp_output_shape(self):
        out = CosineRowNormMLP(D_IN, H, C)(self._input())
        assert out.shape == (N, C)

    def test_cosine_rownorm_mlp_has_no_bias_in_linear_layers(self):
        model = CosineRowNormMLP(D_IN, H, C)
        linear_bias = [n for n, _ in model.named_parameters()
                       if 'bias' in n and 'scale' not in n]
        assert len(linear_bias) == 0, \
            f"CosineRowNormMLP linear layers must have no bias, found: {linear_bias}"

    def test_cosine_rownorm_mlp_has_scale_parameter(self):
        model = CosineRowNormMLP(D_IN, H, C)
        assert hasattr(model, 'scale'), "CosineRowNormMLP must have a scale parameter"
        assert isinstance(model.scale, nn.Parameter)

    # ── LogMagnitudeMLP ──────────────────────────────────────────────────────

    def test_log_magnitude_mlp_output_shape(self):
        out = LogMagnitudeMLP(D_IN, H, C)(self._input())
        assert out.shape == (N, C)

    # ── DualStreamMLP ────────────────────────────────────────────────────────

    def test_dual_stream_mlp_output_shape(self):
        out = DualStreamMLP(D_IN, H, H_MAG, C)(self._input())
        assert out.shape == (N, C)

    def test_dual_stream_mlp_direction_stream_shape(self):
        """Direction branch output must be hidden_dir // 2."""
        model = DualStreamMLP(D_IN, H, H_MAG, C)
        x = self._input()
        M = torch.norm(x, dim=1, keepdim=True)
        x_norm = x / (M + 1e-10)
        h_dir = model.mlp_direction(x_norm)
        assert h_dir.shape == (N, H // 2), \
            f"Direction branch output must be (N, {H // 2}), got {h_dir.shape}"

    def test_dual_stream_mlp_magnitude_stream_shape(self):
        """Magnitude branch output must be hidden_mag."""
        model = DualStreamMLP(D_IN, H, H_MAG, C)
        x = self._input()
        M = torch.norm(x, dim=1, keepdim=True)
        log_M = torch.log(M + 1e-10)
        h_mag = model.mlp_magnitude(log_M)
        assert h_mag.shape == (N, H_MAG), \
            f"Magnitude branch output must be (N, {H_MAG}), got {h_mag.shape}"

    # ── SpectralRowNormMLP ───────────────────────────────────────────────────

    def test_spectral_rownorm_mlp_output_shape(self):
        out = SpectralRowNormMLP(D_IN, H, C, EIGS, alpha=0.5)(self._input())
        assert out.shape == (N, C)

    def test_spectral_rownorm_mlp_alpha_zero_gives_ones_weights(self):
        """alpha=0 must produce all-ones eigenvalue weights (no spectral bias)."""
        model = SpectralRowNormMLP(D_IN, H, C, EIGS, alpha=0.0)
        assert torch.allclose(model.eigenvalue_weights, torch.ones(D_IN)), \
            "alpha=0 eigenvalue weights must be all-ones"

    def test_spectral_rownorm_mlp_negative_alpha(self):
        """Negative alpha must produce finite outputs (no explosion from near-zero eigenvalues)."""
        out = SpectralRowNormMLP(D_IN, H, C, EIGS, alpha=-1.0)(self._input())
        assert torch.isfinite(out).all(), "alpha=-1 must produce finite outputs"

    # ── NestedSpheresMLP ─────────────────────────────────────────────────────

    def test_nested_spheres_mlp_output_shape(self):
        out = NestedSpheresMLP(D_IN, H, C, EIGS, alpha=0.5, beta=1.0)(self._input())
        assert out.shape == (N, C)

    def test_nested_spheres_mlp_alpha_zero(self):
        """alpha=0 must use uniform eigenvalue weights (ones)."""
        model = NestedSpheresMLP(D_IN, H, C, EIGS, alpha=0.0)
        assert torch.allclose(model.eigenvalue_weights, torch.ones(D_IN))

    # ── NestedSpheresClassifier ───────────────────────────────────────────────

    def test_nested_spheres_classifier_output_shape(self):
        out = NestedSpheresClassifier(D_IN, H, C, EIGS, alpha=0.5, beta=1.0)(self._input())
        assert out.shape == (N, C)

    def test_nested_spheres_classifier_beta_zero_no_magnitude(self):
        """beta=0 should zero out the log-magnitude channel — output still finite."""
        out = NestedSpheresClassifier(D_IN, H, C, EIGS, alpha=0.5, beta=0.0)(self._input())
        assert torch.isfinite(out).all()

    # ── Shared: no NaN/Inf ───────────────────────────────────────────────────

    def test_no_nan_or_inf_in_any_model(self):
        """All models must produce finite outputs on random input."""
        x = self._input()
        models = [
            SGC(D_IN, C),
            StandardMLP(D_IN, H, C),
            RowNormMLP(D_IN, H, C),
            CosineRowNormMLP(D_IN, H, C),
            LogMagnitudeMLP(D_IN, H, C),
            DualStreamMLP(D_IN, H, H_MAG, C),
            SpectralRowNormMLP(D_IN, H, C, EIGS, alpha=0.5),
            NestedSpheresMLP(D_IN, H, C, EIGS, alpha=0.5, beta=1.0),
            NestedSpheresClassifier(D_IN, H, C, EIGS, alpha=0.5, beta=1.0),
        ]
        for model in models:
            out = model(x)
            assert torch.isfinite(out).all(), \
                f"{model.__class__.__name__} produced NaN or Inf in output"


# ============================================================================
# 9. train_simple and train_with_selection
# ============================================================================

def _make_loader(X, y, batch_size=16):
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@pytest.fixture(scope="module")
def small_classification_data():
    """Tiny 2-class dataset for training function tests."""
    torch.manual_seed(0)
    np.random.seed(0)
    n, d, c = 60, 8, 2
    X = np.random.randn(n, d).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    X_tr, y_tr = X[:40], y[:40]
    X_va, y_va = X[40:50], y[40:50]
    X_te, y_te = X[50:], y[50:]
    loader = _make_loader(X_tr, y_tr)
    X_va_t = torch.FloatTensor(X_va)
    y_va_t = torch.LongTensor(y_va)
    X_te_t = torch.FloatTensor(X_te)
    y_te_t = torch.LongTensor(y_te)
    return loader, X_va_t, y_va_t, X_te_t, y_te_t, d, c


class TestTrainSimple:

    def test_returns_dict_with_required_keys(self, small_classification_data):
        loader, X_va, y_va, X_te, y_te, d, c = small_classification_data
        model = StandardMLP(d, 16, c)
        result = train_simple(model, loader, X_va, y_va, X_te, y_te,
                              epochs=5, verbose=False)
        assert 'model' in result
        assert 'train_losses' in result
        assert 'val_accs' in result
        assert 'test_acc' in result

    def test_train_losses_length_equals_epochs(self, small_classification_data):
        loader, X_va, y_va, X_te, y_te, d, c = small_classification_data
        model = StandardMLP(d, 16, c)
        result = train_simple(model, loader, X_va, y_va, X_te, y_te,
                              epochs=5, verbose=False)
        assert len(result['train_losses']) == 5
        assert len(result['val_accs']) == 5

    def test_test_acc_in_0_1(self, small_classification_data):
        loader, X_va, y_va, X_te, y_te, d, c = small_classification_data
        model = StandardMLP(d, 16, c)
        result = train_simple(model, loader, X_va, y_va, X_te, y_te,
                              epochs=5, verbose=False)
        assert 0.0 <= result['test_acc'] <= 1.0

    def test_returned_model_is_nn_module(self, small_classification_data):
        loader, X_va, y_va, X_te, y_te, d, c = small_classification_data
        model = StandardMLP(d, 16, c)
        result = train_simple(model, loader, X_va, y_va, X_te, y_te,
                              epochs=3, verbose=False)
        assert isinstance(result['model'], nn.Module)


class TestTrainWithSelection:

    def test_returns_dict_with_required_keys(self, small_classification_data):
        loader, X_va, y_va, X_te, y_te, d, c = small_classification_data
        model = StandardMLP(d, 16, c)
        result = train_with_selection(model, loader, X_va, y_va, X_te, y_te,
                                      epochs=5, verbose=False)
        required = {
            'model', 'train_losses', 'val_losses', 'val_accs',
            'best_val_loss', 'best_val_acc',
            'best_epoch_loss', 'best_epoch_acc',
            'test_at_best_val_loss', 'test_at_best_val_acc'
        }
        assert required.issubset(result.keys())

    def test_best_val_acc_ge_all_val_accs(self, small_classification_data):
        loader, X_va, y_va, X_te, y_te, d, c = small_classification_data
        model = StandardMLP(d, 16, c)
        result = train_with_selection(model, loader, X_va, y_va, X_te, y_te,
                                      epochs=10, verbose=False)
        assert result['best_val_acc'] >= max(result['val_accs']) - 1e-6

    def test_best_val_loss_le_all_val_losses(self, small_classification_data):
        loader, X_va, y_va, X_te, y_te, d, c = small_classification_data
        model = StandardMLP(d, 16, c)
        result = train_with_selection(model, loader, X_va, y_va, X_te, y_te,
                                      epochs=10, verbose=False)
        assert result['best_val_loss'] <= min(result['val_losses']) + 1e-6

    def test_test_accs_in_0_1(self, small_classification_data):
        loader, X_va, y_va, X_te, y_te, d, c = small_classification_data
        model = StandardMLP(d, 16, c)
        result = train_with_selection(model, loader, X_va, y_va, X_te, y_te,
                                      epochs=5, verbose=False)
        assert 0.0 <= result['test_at_best_val_loss'] <= 1.0
        assert 0.0 <= result['test_at_best_val_acc'] <= 1.0

    def test_epoch_indices_within_range(self, small_classification_data):
        loader, X_va, y_va, X_te, y_te, d, c = small_classification_data
        model = StandardMLP(d, 16, c)
        result = train_with_selection(model, loader, X_va, y_va, X_te, y_te,
                                      epochs=8, verbose=False)
        assert 0 <= result['best_epoch_loss'] < 8
        assert 0 <= result['best_epoch_acc'] < 8
