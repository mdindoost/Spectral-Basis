#!/bin/bash
# =============================================================================
# Run All Whitening Experiments
# =============================================================================
#
# This script runs investigation_whitening_rownorm.py on all datasets
# with both fixed and random splits, for k=0, 2, 10
#
# Total experiments: 9 datasets × 2 split types × 3 k values = 54 experiments
#
# Usage:
#   chmod +x scripts/run_all_whitening_experiments.sh
#   ./scripts/run_all_whitening_experiments.sh
#
# Or run specific parts:
#   ./scripts/run_all_whitening_experiments.sh fixed    # Only fixed splits
#   ./scripts/run_all_whitening_experiments.sh random   # Only random splits
#
# =============================================================================

set -e  # Exit on error

# Configuration
DATASETS="cora citeseer pubmed wikics amazon-photo amazon-computers coauthor-cs coauthor-physics ogbn-arxiv"
K_VALUES="0 2 10"
SCRIPT="experiments/investigation_whitening_rownorm.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
SPLIT_FILTER=${1:-"all"}  # "fixed", "random", or "all"

echo "=============================================================================="
echo "WHITENING EXPERIMENTS: Does RowNorm Fix Standard Whitening?"
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  Datasets: $DATASETS"
echo "  K values: $K_VALUES"
echo "  Split filter: $SPLIT_FILTER"
echo "  Whitening eps: 1e-6 (proper whitening)"
echo ""

# Create log directory
LOG_DIR="logs/whitening_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Track progress
TOTAL=0
COMPLETED=0
FAILED=0

# Count total experiments
for dataset in $DATASETS; do
    for k in $K_VALUES; do
        if [ "$SPLIT_FILTER" = "all" ] || [ "$SPLIT_FILTER" = "fixed" ]; then
            TOTAL=$((TOTAL + 1))
        fi
        if [ "$SPLIT_FILTER" = "all" ] || [ "$SPLIT_FILTER" = "random" ]; then
            TOTAL=$((TOTAL + 1))
        fi
    done
done

echo "Total experiments to run: $TOTAL"
echo ""
echo "=============================================================================="
echo "Starting experiments..."
echo "=============================================================================="

# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local splits=$2
    local k=$3
    local exp_num=$4

    local log_file="$LOG_DIR/${dataset}_${splits}_k${k}.log"

    echo -e "${BLUE}[$exp_num/$TOTAL]${NC} Running: $dataset (splits=$splits, k=$k)"

    if python $SCRIPT $dataset --splits $splits --component lcc --k_diffusion $k > "$log_file" 2>&1; then
        echo -e "  ${GREEN}✓ Completed${NC}"
        return 0
    else
        echo -e "  ${RED}✗ Failed${NC} (see $log_file)"
        return 1
    fi
}

# Run experiments
EXP_NUM=0

# Fixed splits
if [ "$SPLIT_FILTER" = "all" ] || [ "$SPLIT_FILTER" = "fixed" ]; then
    echo ""
    echo -e "${YELLOW}=== FIXED SPLITS ===${NC}"
    echo ""

    for dataset in $DATASETS; do
        for k in $K_VALUES; do
            EXP_NUM=$((EXP_NUM + 1))
            if run_experiment $dataset "fixed" $k $EXP_NUM; then
                COMPLETED=$((COMPLETED + 1))
            else
                FAILED=$((FAILED + 1))
            fi
        done
    done
fi

# Random splits
if [ "$SPLIT_FILTER" = "all" ] || [ "$SPLIT_FILTER" = "random" ]; then
    echo ""
    echo -e "${YELLOW}=== RANDOM SPLITS ===${NC}"
    echo ""

    for dataset in $DATASETS; do
        for k in $K_VALUES; do
            EXP_NUM=$((EXP_NUM + 1))
            if run_experiment $dataset "random" $k $EXP_NUM; then
                COMPLETED=$((COMPLETED + 1))
            else
                FAILED=$((FAILED + 1))
            fi
        done
    done
fi

# Summary
echo ""
echo "=============================================================================="
echo "EXPERIMENT SUMMARY"
echo "=============================================================================="
echo -e "  Total:     $TOTAL"
echo -e "  ${GREEN}Completed:${NC} $COMPLETED"
echo -e "  ${RED}Failed:${NC}    $FAILED"
echo ""
echo "Logs saved to: $LOG_DIR"
echo ""

# Run analysis if all completed
if [ $FAILED -eq 0 ]; then
    echo "=============================================================================="
    echo "Running analysis..."
    echo "=============================================================================="

    if [ "$SPLIT_FILTER" = "fixed" ]; then
        python scripts/analyze_whitening_rownorm.py --splits fixed
    elif [ "$SPLIT_FILTER" = "random" ]; then
        python scripts/analyze_whitening_rownorm.py --splits random
    else
        echo ""
        echo "=== FIXED SPLITS ANALYSIS ==="
        python scripts/analyze_whitening_rownorm.py --splits fixed

        echo ""
        echo "=== RANDOM SPLITS ANALYSIS ==="
        python scripts/analyze_whitening_rownorm.py --splits random
    fi
else
    echo -e "${RED}Some experiments failed. Check logs for details.${NC}"
    echo "After fixing issues, run analysis with:"
    echo "  python scripts/analyze_whitening_rownorm.py --splits fixed"
    echo "  python scripts/analyze_whitening_rownorm.py --splits random"
fi

echo ""
echo "=============================================================================="
echo "DONE"
echo "=============================================================================="
