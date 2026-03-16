#!/bin/bash
cd /home/md724/Spectral-Basis

PYTHON=venv/bin/python
DATASETS="cora citeseer pubmed ogbn-arxiv wikics amazon-computers amazon-photo coauthor-cs coauthor-physics"
K_VALUES="1 2 4 6 8 10 12 20 30"

for ds in $DATASETS; do
    echo "======== $ds ========"
    $PYTHON PAPER_EXPERIMENTS/exp9_softmax_convergence.py $ds --k_values $K_VALUES
done

echo "======== ALL DONE ========"
