#!/bin/bash

# Run Random Subspace Experiment on All Datasets
# ===============================================
# This script runs the random subspace experiment sequentially on all datasets

# Usage:
#   ./scripts/run_random_all_datasets.sh              # Fixed splits (default)
#   ./scripts/run_random_all_datasets.sh --random-splits  # Random 60/20/20 splits

# Check for --random-splits flag
RANDOM_SPLITS_FLAG=""
if [[ "$1" == "--random-splits" ]]; then
    RANDOM_SPLITS_FLAG="--random-splits"
    SPLIT_TYPE="random 60/20/20 splits"
    ESTIMATED_TIME="6-18 hours"
else
    SPLIT_TYPE="fixed benchmark splits"
    ESTIMATED_TIME="1-3 hours"
fi

echo "========================================================================"
echo "RANDOM SUBSPACE EXPERIMENT: BATCH RUN"
echo "========================================================================"
echo "Running on all 9 datasets"
echo "Split type: $SPLIT_TYPE"
echo "Estimated time: $ESTIMATED_TIME"
echo "========================================================================"
echo ""

# Array of datasets
DATASETS=(
    "ogbn-arxiv"
    "cora"
    "citeseer"
    "pubmed"
    "wikics"
    "amazon-photo"
    "amazon-computers"
    "coauthor-cs"
    "coauthor-physics"
)

# Track timing
TOTAL_START=$(date +%s)
SUCCESS_COUNT=0
FAIL_COUNT=0

# Run each dataset
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "STARTING: $dataset"
    echo "========================================================================"
    
    START=$(date +%s)
    
    # Run with or without --random-splits flag
    if python experiments/investigation2_random_subspaces.py "$dataset" $RANDOM_SPLITS_FLAG; then
        END=$(date +%s)
        DURATION=$((END - START))
        MINUTES=$((DURATION / 60))
        SECONDS=$((DURATION % 60))
        
        echo ""
        echo "✓ COMPLETED: $dataset in ${MINUTES}m ${SECONDS}s"
        echo ""
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        END=$(date +%s)
        DURATION=$((END - START))
        MINUTES=$((DURATION / 60))
        SECONDS=$((DURATION % 60))
        
        echo ""
        echo "✗ FAILED: $dataset after ${MINUTES}m ${SECONDS}s"
        echo ""
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# Summary
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "========================================================================"
echo "BATCH RUN COMPLETE"
echo "========================================================================"
echo "Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "Successful: $SUCCESS_COUNT/${#DATASETS[@]}"
echo "Failed: $FAIL_COUNT/${#DATASETS[@]}"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "Running cross-dataset summary..."
    echo ""
    
    if [[ "$RANDOM_SPLITS_FLAG" == "--random-splits" ]]; then
        python scripts/summarize_random_subspaces.py --random-splits
    else
        python scripts/summarize_random_subspaces.py
    fi
else
    echo "No successful runs to summarize."
fi

echo ""
echo "========================================================================"
if [ $SUCCESS_COUNT -eq ${#DATASETS[@]} ]; then
    echo "✓ ALL EXPERIMENTS COMPLETE"
elif [ $SUCCESS_COUNT -gt 0 ]; then
    echo "⚠️  PARTIAL SUCCESS"
else
    echo "✗ ALL EXPERIMENTS FAILED"
fi
echo "========================================================================"
echo "Results location:"
if [[ "$RANDOM_SPLITS_FLAG" == "--random-splits" ]]; then
    echo "  results/investigation2_random_subspaces/<dataset>/random_splits/"
else
    echo "  results/investigation2_random_subspaces/<dataset>/fixed_splits/"
fi
echo "========================================================================"