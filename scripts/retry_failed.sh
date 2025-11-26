#!/bin/bash

################################################################################
# Retry Failed Experiments
################################################################################

LOG_DIR="logs/parallel_experiments"
SCRIPT_PATH="experiments/investigation_sgc_antiSmoothing_partAB.py"
K_VALUES="2 4 6 8 10"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================================================="
echo "RETRY FAILED EXPERIMENTS"
echo "============================================================================="
echo ""

# Find failed experiments
FAILED_EXPERIMENTS=()

for status_file in "$LOG_DIR"/*.status; do
    if [ -f "$status_file" ]; then
        status=$(cat "$status_file")
        if [ "$status" = "FAILED" ]; then
            exp_name=$(basename "$status_file" .status)
            FAILED_EXPERIMENTS+=("$exp_name")
        fi
    fi
done

if [ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ No failed experiments found!${NC}"
    exit 0
fi

echo -e "${YELLOW}Found ${#FAILED_EXPERIMENTS[@]} failed experiments:${NC}"
for exp in "${FAILED_EXPERIMENTS[@]}"; do
    echo "  • $exp"
done
echo ""

read -p "Retry these experiments? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Retrying failed experiments..."
echo ""

for exp in "${FAILED_EXPERIMENTS[@]}"; do
    # Parse dataset and split_type
    dataset=$(echo "$exp" | sed 's/_fixed$//' | sed 's/_random$//')
    split_type="fixed"
    if [[ "$exp" == *"_random" ]]; then
        split_type="random"
    fi
    
    log_file="$LOG_DIR/${exp}.log"
    
    echo -e "${YELLOW}→ Retrying: $dataset ($split_type)${NC}"
    
    # Remove old status
    rm -f "$LOG_DIR/${exp}.status"
    
    # Run experiment
    python "$SCRIPT_PATH" "$dataset" \
        --k_diffusion $K_VALUES \
        --splits "$split_type" \
        --component lcc \
        > "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ SUCCESS: $dataset ($split_type)${NC}"
        echo "COMPLETED" > "$LOG_DIR/${exp}.status"
    else
        echo -e "${RED}✗ FAILED AGAIN: $dataset ($split_type)${NC}"
        echo "FAILED" > "$LOG_DIR/${exp}.status"
    fi
    echo ""
done

echo "============================================================================="
echo "RETRY COMPLETE"
echo "============================================================================="
