# MLP Sensitivity to Spectral Basis Representations

Research investigating how MLPs respond to different spectral basis representations of graph-structured data.

## Investigations

### Investigation 1: True Eigenvectors
- Compute true graph eigenvectors via `eigsh(L, k=2×num_classes, M=D)`
- Compare 6 different models/preprocessing approaches
- Finding: RowNorm MLP significantly outperforms Standard MLP (~5-10%)

### Investigation 2: X-Restricted Eigenvectors  
- Solve restricted eigenproblem: `(X^T L X)v = λ(X^T D X)v`
- Key insight: span(U) = span(X) — same information, different basis
- Compare: Raw X (scaled) vs Restricted U (row-normalized)
- Finding: Small but real improvement (~1-3%)

## Running Experiments
```bash
