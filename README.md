# MLP Sensitivity to Spectral Basis Representations

Research investigating how MLPs respond to different spectral basis representations of graph-structured data.



## Tests

A test suite for the core utility functions lives in `src/test_utils.py`.
It covers 46 tests across all functions in `src/utils.py`, including the
critical matrix-convention checks that guard against the L/D swap bug.

**Run the tests:**
```bash
venv/bin/python -m pytest src/test_utils.py -v
```

**What is tested:**

| Area | Key checks |
|------|-----------|
| `build_graph_matrices` | Return order (adj, L, D), symmetry, self-loops, L = D − A, PSD, known degrees |
| `compute_restricted_eigenvectors` | D-orthonormality, eigenvalues in [0, 2], Rayleigh quotient, swap alarm fires, subspace constraint |
| `get_largest_connected_component_nx` | Connected/disconnected graphs, correct LCC selection |
| `extract_subgraph` | Shape, feature/label rows, split index remapping |
| `compute_sgc_normalized_adjacency` | Symmetry, eigenvalues in [−1, 1] |
| `sgc_precompute` | degree=0/1/2 correctness, shape preserved |
| Model architectures | Output shapes, no NaN/Inf, RowNorm has no bias, alpha=0 weights |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{spectral-basis-mlp-2024,
  author = {Dindoost, Mohammad},
  title = {MLP Sensitivity to Spectral Basis Representations},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mdindoost/Spectral-Basis}}
}
```

---

## Contact

**Mohammad Dindoost**  
PhD Candidate, New Jersey Institute of Technology  
Email: md724@njit.edu  
GitHub: [@mdindoost](https://github.com/mdindoost)


---

## License

MIT License - See LICENSE file for details

---


**First Updated**: May - 2025
