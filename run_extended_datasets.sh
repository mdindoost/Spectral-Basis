#!/bin/bash

# Run Investigation 2 Directions A&B on Extended Datasets
# ========================================================
# This script runs experiments on larger graphs to validate findings

echo "=================================================="
echo "Investigation 2 Directions A&B: Extended Datasets"
echo "=================================================="
echo ""
echo "This will run on 5 additional larger datasets:"
echo "  1. WikiCS           (~11K nodes, 10 classes)"
echo "  2. Amazon-Photo     (~7.6K nodes, 8 classes)"
echo "  3. Amazon-Computers (~13.7K nodes, 10 classes)"
echo "  4. Coauthor-CS      (~18K nodes, 15 classes)"
echo "  5. Coauthor-Physics (~34K nodes, 5 classes)"
echo ""
echo "Estimated total runtime: 4-6 hours"
echo "=================================================="
echo ""

# Ask for confirmation
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# Create log directory
mkdir -p logs

# Array of datasets
DATASETS=("wikics" "amazon-photo" "amazon-computers" "coauthor-cs" "coauthor-physics")

# Run each dataset
for dataset in "${DATASETS[@]}"
do
    echo ""
    echo "=================================================="
    echo "Running: $dataset"
    echo "=================================================="
    
    # Run with fixed splits
    echo "Running with fixed benchmark splits..."
    python experiments/investigation2_directions_AB.py $dataset 2>&1 | tee logs/${dataset}_fixed.log
    
    # Optionally run with random splits (uncomment if needed)
    echo "Running with random 60/20/20 splits..."
    python experiments/investigation2_directions_AB.py $dataset --random-splits 2>&1 | tee logs/${dataset}_random.log
    
    echo "✓ Completed: $dataset"
    echo ""
done

echo "=================================================="
echo "✓ All datasets completed!"
echo "=================================================="
echo ""
echo "Results saved in:"
echo "  results/investigation2_directions_AB/<dataset>/fixed_splits/"
echo ""
echo "Logs saved in:"
echo "  logs/<dataset>_fixed.log"
echo ""
echo "To view summary of all results, run:"
echo "  python scripts/summarize_results.py"
