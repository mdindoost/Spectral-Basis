#!/bin/bash
set -e

PYTHON=/home/md724/Spectral-Basis/venv/bin/python
SCRIPTS=/home/md724/Spectral-Basis/PAPER_EXPERIMENTS

echo "========================================================"
echo " FULL PAPER PIPELINE"
echo " Start: $(date)"
echo "========================================================"

# ── Step 1: Training + Analytics per dataset ─────────────────────────────────
for ds in cora citeseer amazon-photo wikics amazon-computers coauthor-cs pubmed coauthor-physics ogbn-arxiv; do

    echo ""
    echo "--------------------------------------------------------"
    echo " DATASET: $ds  |  $(date)"
    echo "--------------------------------------------------------"

    echo "[1/2] Training: $ds fixed"
    $PYTHON $SCRIPTS/master_training.py $ds --splits fixed

    echo "[2/2] Analytics: $ds fixed"
    $PYTHON $SCRIPTS/master_analytics.py $ds --splits fixed

    echo "[1/2] Training: $ds random"
    $PYTHON $SCRIPTS/master_training.py $ds --splits random

    echo "[2/2] Analytics: $ds random"
    $PYTHON $SCRIPTS/master_analytics.py $ds --splits random

    echo "  ✓ $ds complete  |  $(date)"
done

# ── Step 2: Generate all paper artifacts ─────────────────────────────────────
echo ""
echo "========================================================"
echo " GENERATING PAPER ARTIFACTS  |  $(date)"
echo "========================================================"
$PYTHON $SCRIPTS/generate_paper_artifacts_v2.py --all

echo ""
echo "========================================================"
echo " ALL DONE  |  $(date)"
echo "========================================================"
