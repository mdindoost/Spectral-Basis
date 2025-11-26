#!/bin/bash

################################################################################
# Check Specific Experiment Status
################################################################################

if [ $# -ne 2 ]; then
    echo "Usage: $0 <dataset> <split_type>"
    echo ""
    echo "Examples:"
    echo "  $0 ogbn-arxiv fixed"
    echo "  $0 cora random"
    exit 1
fi

DATASET=$1
SPLIT_TYPE=$2

LOG_DIR="logs/parallel_experiments"
RESULTS_DIR="results/investigation_sgc_antiSmoothing"

LOG_FILE="$LOG_DIR/${DATASET}_${SPLIT_TYPE}.log"
STATUS_FILE="$LOG_DIR/${DATASET}_${SPLIT_TYPE}.status"
RESULT_DIR="$RESULTS_DIR/${DATASET}_${SPLIT_TYPE}_lcc"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "============================================================================="
echo "EXPERIMENT STATUS: $DATASET ($SPLIT_TYPE)"
echo "============================================================================="
echo ""

# Check status file
if [ -f "$STATUS_FILE" ]; then
    status=$(cat "$STATUS_FILE")
    if [ "$status" = "COMPLETED" ]; then
        echo -e "${GREEN}Status: COMPLETED ✓${NC}"
    elif [ "$status" = "FAILED" ]; then
        echo -e "${RED}Status: FAILED ✗${NC}"
    fi
else
    echo -e "${YELLOW}Status: RUNNING or NOT STARTED${NC}"
fi
echo ""

# Check log file
if [ -f "$LOG_FILE" ]; then
    echo "Log file: $LOG_FILE"
    echo "Log size: $(du -h "$LOG_FILE" | cut -f1)"
    echo "Last modified: $(date -r "$LOG_FILE" '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    echo "--- Current progress (last 30 lines) ---"
    tail -30 "$LOG_FILE" | grep -E "DIFFUSION k=|Experiment:|COMPLETE|error|Error|ERROR" | tail -10
    echo ""
    
    # Check for errors
    if grep -q -i "error\|exception\|traceback" "$LOG_FILE"; then
        echo -e "${RED}⚠ Errors detected in log!${NC}"
        echo ""
        echo "--- Last error (if any) ---"
        grep -i "error\|exception" "$LOG_FILE" | tail -5
        echo ""
    fi
else
    echo -e "${YELLOW}⚠ Log file not found${NC}"
    echo ""
fi

# Check results
if [ -d "$RESULT_DIR" ]; then
    echo "Results directory: $RESULT_DIR"
    echo ""
    echo "Available k values:"
    for k_dir in "$RESULT_DIR"/k*/; do
        if [ -d "$k_dir" ]; then
            k=$(basename "$k_dir")
            result_file="$k_dir/metrics/results.json"
            if [ -f "$result_file" ]; then
                # Extract test accuracy from JSON
                acc=$(python3 -c "import json; data=json.load(open('$result_file')); print(f\"{data['results']['full_rownorm_mlp']['test_acc_mean']*100:.2f}%\")" 2>/dev/null)
                echo -e "  ${GREEN}✓${NC} $k: Full+RowNorm = $acc"
            else
                echo -e "  ${YELLOW}⊘${NC} $k: No results file"
            fi
        fi
    done
else
    echo -e "${YELLOW}⚠ Results directory not found${NC}"
fi

echo ""
echo "============================================================================="

# Offer to show full log
echo ""
read -p "Show full log? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "$LOG_FILE" ]; then
        less "$LOG_FILE"
    else
        echo "Log file not found"
    fi
fi
