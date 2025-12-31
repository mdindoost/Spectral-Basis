#!/bin/bash
# Run whitening experiments with max 3 parallel jobs
# Usage: bash scripts/run_whitening_experiments.sh

MAX_JOBS=3
DATASETS="ogbn-arxiv cora citeseer pubmed wikics amazon-photo amazon-computers coauthor-cs coauthor-physics"
K_VALUES="0 2 10"

LOG_DIR="logs/whitening_experiments"
mkdir -p "$LOG_DIR"

# Function to wait if we have too many jobs
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 5
    done
}

echo "========================================"
echo "Running Whitening Experiments"
echo "Max parallel jobs: $MAX_JOBS"
echo "Datasets: $DATASETS"
echo "K values: $K_VALUES"
echo "========================================"

# Count total jobs
TOTAL=0
for dataset in $DATASETS; do
    for k in $K_VALUES; do
        ((TOTAL++))
    done
done
echo "Total experiments: $TOTAL"
echo "========================================"

# Run experiments
STARTED=0
for dataset in $DATASETS; do
    for k in $K_VALUES; do
        wait_for_slot

        ((STARTED++))
        LOGFILE="$LOG_DIR/${dataset}_k${k}.log"

        echo "[$STARTED/$TOTAL] Starting: $dataset k=$k"

        python experiments/investigation_whitening_rownorm.py "$dataset" \
            --splits fixed \
            --component lcc \
            --k_diffusion "$k" \
            > "$LOGFILE" 2>&1 &

        # Small delay to avoid race conditions
        sleep 1
    done
done

# Wait for all remaining jobs
echo ""
echo "Waiting for remaining jobs to complete..."
wait

echo ""
echo "========================================"
echo "All experiments complete!"
echo "Logs saved to: $LOG_DIR/"
echo "========================================"
echo ""
echo "Run analysis with:"
echo "  python scripts/analyze_whitening_rownorm.py"
