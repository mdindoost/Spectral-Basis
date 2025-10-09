#!/bin/bash
# Run all experiments for all datasets

echo "======================================"
echo "Running Spectral Basis Experiments"
echo "======================================"

# Dataset list
DATASETS=("ogbn-arxiv" "cora" "citeseer" "pubmed")

# Investigation 1: True Eigenvectors
echo ""
echo "=== INVESTIGATION 1: TRUE EIGENVECTORS ==="
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Running Investigation 1 on $dataset..."
    python experiments/investigation1_true_eigenvectors.py "$dataset"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Investigation 1 failed for $dataset"
        exit 1
    fi
done

# Investigation 2: X-Restricted Eigenvectors
echo ""
echo "=== INVESTIGATION 2: X-RESTRICTED EIGENVECTORS ==="
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Running Investigation 2 on $dataset..."
    python experiments/investigation2_restricted_eigenvectors.py "$dataset"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Investigation 2 failed for $dataset"
        exit 1
    fi
done

echo ""
echo "======================================"
echo "All experiments completed successfully!"
echo "======================================"
echo ""
echo "Results saved to:"
for dataset in "${DATASETS[@]}"; do
    echo "  - results/investigation1/$dataset/"
    echo "  - results/investigation2/$dataset/"
done
