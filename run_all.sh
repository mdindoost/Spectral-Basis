#!/bin/bash
set -e

PYTHON=/home/md724/Spectral-Basis/venv/bin/python
SCRIPTS=/home/md724/Spectral-Basis/PAPER_EXPERIMENTS

for ds in cora citeseer amazon-photo wikics amazon-computers coauthor-cs pubmed coauthor-physics ogbn-arxiv; do
    echo "=== $ds fixed ==="
    $PYTHON $SCRIPTS/master_training.py $ds --splits fixed
    $PYTHON $SCRIPTS/master_analytics.py $ds --splits fixed

    echo "=== $ds random ==="
    $PYTHON $SCRIPTS/master_training.py $ds --splits random
    $PYTHON $SCRIPTS/master_analytics.py $ds --splits random
done

echo "=== ALL DONE ==="
