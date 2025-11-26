#!/bin/bash

################################################################################
# Real-time Progress Monitor for Parallel Experiments
################################################################################

# Configuration
LOG_DIR="logs/parallel_experiments"
RESULTS_DIR="results/investigation_sgc_antiSmoothing"
REFRESH_INTERVAL=10  # seconds

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Datasets
DATASETS=(
    "ogbn-arxiv"
    "wikics"
    "amazon-computers"
    "amazon-photo"
    "coauthor-cs"
    "coauthor-physics"
    "cora"
    "citeseer"
    "pubmed"
)

SPLIT_TYPES=("fixed" "random")

################################################################################
# Functions
################################################################################

get_status() {
    local dataset=$1
    local split_type=$2
    local status_file="$LOG_DIR/${dataset}_${split_type}.status"
    local log_file="$LOG_DIR/${dataset}_${split_type}.log"
    local result_file="$RESULTS_DIR/${dataset}_${split_type}_lcc/k10/metrics/results.json"
    
    if [ -f "$status_file" ]; then
        status=$(cat "$status_file")
        if [ "$status" = "COMPLETED" ]; then
            echo -e "${GREEN}✓ DONE${NC}"
        elif [ "$status" = "FAILED" ]; then
            echo -e "${RED}✗ FAIL${NC}"
        fi
    elif [ -f "$log_file" ]; then
        # Check what's happening in the log
        if tail -20 "$log_file" | grep -q "EXPERIMENT COMPLETE"; then
            echo -e "${GREEN}✓ DONE${NC}"
        elif tail -20 "$log_file" | grep -q "DIFFUSION k="; then
            current_k=$(tail -50 "$log_file" | grep "DIFFUSION k=" | tail -1 | grep -oP 'k=\K[0-9]+')
            echo -e "${CYAN}→ k=$current_k${NC}"
        elif tail -20 "$log_file" | grep -q "Loading dataset"; then
            echo -e "${YELLOW}⟳ LOAD${NC}"
        else
            echo -e "${YELLOW}⟳ RUN${NC}"
        fi
    elif [ -f "$result_file" ]; then
        echo -e "${GREEN}✓ SKIP${NC}"
    else
        echo -e "  WAIT"
    fi
}

get_gpu_usage() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1
    else
        echo "N/A"
    fi
}

################################################################################
# Main Monitor Loop
################################################################################

clear

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}                    EXPERIMENT PROGRESS MONITOR${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""
echo "Press Ctrl+C to exit"
echo "Refreshing every ${REFRESH_INTERVAL}s..."
echo ""

while true; do
    # Get current time
    CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Move cursor to top (for live updates)
    tput cup 5 0
    
    echo -e "${CYAN}Last update: $CURRENT_TIME${NC}"
    echo ""
    
    # GPU status
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(get_gpu_usage)
        echo -e "${YELLOW}GPU Status:${NC} $gpu_info"
    else
        echo -e "${YELLOW}GPU Status:${NC} nvidia-smi not available"
    fi
    echo ""
    
    # Count statuses
    COMPLETED=0
    RUNNING=0
    FAILED=0
    WAITING=0
    SKIPPED=0
    
    # Print table header
    printf "%-20s %-12s %-12s\n" "Dataset" "Fixed" "Random"
    echo "---------------------------------------------------"
    
    # Print status for each dataset
    for dataset in "${DATASETS[@]}"; do
        printf "%-20s " "$dataset"
        
        for split_type in "${SPLIT_TYPES[@]}"; do
            status=$(get_status "$dataset" "$split_type")
            printf "%-12s " "$status"
            
            # Count for summary
            if [[ "$status" == *"DONE"* ]]; then
                ((COMPLETED++))
            elif [[ "$status" == *"FAIL"* ]]; then
                ((FAILED++))
            elif [[ "$status" == *"RUN"* ]] || [[ "$status" == *"k="* ]] || [[ "$status" == *"LOAD"* ]]; then
                ((RUNNING++))
            elif [[ "$status" == *"SKIP"* ]]; then
                ((SKIPPED++))
            else
                ((WAITING++))
            fi
        done
        echo ""
    done
    
    echo ""
    echo "---------------------------------------------------"
    echo "Summary:"
    echo -e "  ${GREEN}✓ Completed:${NC} $COMPLETED"
    echo -e "  ${CYAN}→ Running:${NC} $RUNNING"
    echo -e "  ${RED}✗ Failed:${NC} $FAILED"
    echo -e "  ${YELLOW}⊘ Skipped:${NC} $SKIPPED"
    echo -e "    Waiting: $WAITING"
    
    TOTAL=$((${#DATASETS[@]} * ${#SPLIT_TYPES[@]}))
    PROGRESS=$(( (COMPLETED + SKIPPED) * 100 / TOTAL ))
    echo ""
    echo -e "Progress: ${PROGRESS}% ($((COMPLETED + SKIPPED))/$TOTAL)"
    
    # Progress bar
    echo -n "["
    for i in {1..50}; do
        if [ $i -le $((PROGRESS / 2)) ]; then
            echo -n "="
        else
            echo -n " "
        fi
    done
    echo "]"
    
    echo ""
    echo "---------------------------------------------------"
    echo ""
    
    # Check for recent completions
    echo "Recent activity (last 5 minutes):"
    find "$LOG_DIR" -name "*.log" -mmin -5 -exec basename {} .log \; 2>/dev/null | head -5 | while read exp; do
        echo "  • $exp"
    done
    
    echo ""
    echo "                                                    "  # Clear any leftover text
    
    # Wait before next refresh
    sleep $REFRESH_INTERVAL
done
