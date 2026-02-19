#!/bin/bash
# =============================================================================
# Run Remaining Whitening Experiments (Continue from where cancelled)
# =============================================================================
#
# Status after cancellation:
#   Fixed splits DONE: cora, citeseer, pubmed, wikics, amazon-photo,
#                      amazon-computers, coauthor-cs
#   Fixed splits TODO: coauthor-physics, ogbn-arxiv
#   Random splits TODO: ALL (old results have eps=1e-3, need eps=1e-6)
#
# Usage:
#   ./scripts/run_remaining_experiments.sh           # Run all remaining
#   ./scripts/run_remaining_experiments.sh fixed     # Only remaining fixed
#   ./scripts/run_remaining_experiments.sh random    # Only random splits
#
# =============================================================================

set -e

SCRIPT="experiments/investigation_whitening_rownorm.py"
K_VALUES="0 2 10"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SPLIT_FILTER=${1:-"all"}

echo "=============================================================================="
echo "CONTINUING WHITENING EXPERIMENTS (eps=1e-6)"
echo "=============================================================================="

LOG_DIR="logs/whitening_remaining_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs: $LOG_DIR"
echo ""

run_exp() {
    local dataset=$1
    local splits=$2
    local k=$3
    echo -e "${BLUE}Running:${NC} $dataset (splits=$splits, k=$k)"
    python $SCRIPT $dataset --splits $splits --component lcc --k_diffusion $k \
        > "$LOG_DIR/${dataset}_${splits}_k${k}.log" 2>&1
    echo -e "  ${GREEN}✓ Done${NC}"
}

# =============================================================================
# REMAINING FIXED SPLITS
# =============================================================================
if [ "$SPLIT_FILTER" = "all" ] || [ "$SPLIT_FILTER" = "fixed" ]; then
    echo -e "${YELLOW}=== REMAINING FIXED SPLITS ===${NC}"
    echo ""

    # coauthor-physics (was cancelled here)
    for k in $K_VALUES; do
        run_exp "coauthor-physics" "fixed" $k
    done

    # ogbn-arxiv
    for k in $K_VALUES; do
        run_exp "ogbn-arxiv" "fixed" $k
    done

    echo ""
    echo -e "${GREEN}Fixed splits complete!${NC}"
fi

# =============================================================================
# ALL RANDOM SPLITS (need fresh run with eps=1e-6)
# =============================================================================
if [ "$SPLIT_FILTER" = "all" ] || [ "$SPLIT_FILTER" = "random" ]; then
    echo ""
    echo -e "${YELLOW}=== ALL RANDOM SPLITS (re-running with eps=1e-6) ===${NC}"
    echo ""

    DATASETS="cora citeseer pubmed wikics amazon-photo amazon-computers coauthor-cs coauthor-physics ogbn-arxiv"

    for dataset in $DATASETS; do
        for k in $K_VALUES; do
            run_exp $dataset "random" $k
        done
    done

    echo ""
    echo -e "${GREEN}Random splits complete!${NC}"
fi

# =============================================================================
# RUN ANALYSIS
# =============================================================================
echo ""
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

echo ""
echo "=============================================================================="
echo "ALL DONE!"
echo "=============================================================================="
