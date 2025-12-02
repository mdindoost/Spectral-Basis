#!/bin/bash

################################################################################
# INVESTIGATION 5: RUN ALL EXPERIMENTS (FIXED THEN RANDOM)
################################################################################
#
# This script runs Investigation 5 experiments on all datasets:
# 1. First runs all FIXED splits (2 in parallel)
# 2. Then runs all RANDOM splits (2 in parallel)
#
# Usage:
#   bash run_investigation5_all.sh
#
# To run with different parallelism:
#   MAX_PARALLEL=3 bash run_investigation5_all.sh
#
# Author: Mohammad
# Date: December 2025
################################################################################

# Configuration
MAX_PARALLEL=${MAX_PARALLEL:-2}  # Maximum parallel jobs
PYTHON_CMD=${PYTHON_CMD:-python}
SCRIPT_NAME="experiments/investigation5_multiscale_fusion.py"

# Log directory
LOG_DIR="logs/investigation5"
mkdir -p "$LOG_DIR"

# Datasets to run
DATASETS=(
    "ogbn-arxiv"
    "amazon-computers"
    "amazon-photo"
    "coauthor-cs"
    "coauthor-physics"
    "wikics"
    "cora"
    "citeseer"
    "pubmed"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo ""
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local split_type=$2
    local log_file="$LOG_DIR/${dataset}_${split_type}_$(date +%Y%m%d_%H%M%S).log"
    
    print_info "Starting: $dataset ($split_type)"
    print_info "Log file: $log_file"
    
    # Run experiment and capture output
    $PYTHON_CMD $SCRIPT_NAME \
        --dataset "$dataset" \
        --split_type "$split_type" \
        --component_type lcc \
        --num_seeds 5 \
        > "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "Completed: $dataset ($split_type)"
        return 0
    else
        print_error "Failed: $dataset ($split_type) - Exit code: $exit_code"
        print_error "Check log: $log_file"
        return 1
    fi
}

# Function to wait for jobs to finish (keep max parallel)
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 5
    done
}

# Function to wait for all jobs
wait_all() {
    print_info "Waiting for all jobs to complete..."
    wait
    print_success "All jobs completed!"
}

################################################################################
# Main Execution
################################################################################

print_header "INVESTIGATION 5: MULTI-SCALE FUSION - ALL EXPERIMENTS"

print_info "Configuration:"
print_info "  Max parallel jobs: $MAX_PARALLEL"
print_info "  Python command: $PYTHON_CMD"
print_info "  Script: $SCRIPT_NAME"
print_info "  Datasets: ${#DATASETS[@]}"
print_info "  Log directory: $LOG_DIR"

# Check if script exists
if [ ! -f "$SCRIPT_NAME" ]; then
    print_error "Script not found: $SCRIPT_NAME"
    print_error "Please run this script from the directory containing $SCRIPT_NAME"
    exit 1
fi

# Record start time
START_TIME=$(date +%s)

################################################################################
# PHASE 1: FIXED SPLITS
################################################################################

print_header "PHASE 1: RUNNING FIXED SPLITS (${#DATASETS[@]} datasets)"

FIXED_SUCCESS=0
FIXED_FAILED=0

for dataset in "${DATASETS[@]}"; do
    wait_for_slot  # Wait if max parallel jobs reached
    
    # Run in background
    (
        if run_experiment "$dataset" "fixed"; then
            exit 0
        else
            exit 1
        fi
    ) &
    
    # Store PID for tracking
    PIDS+=($!)
    
    sleep 2  # Small delay between launches
done

# Wait for all fixed splits to complete
print_info "Waiting for all FIXED splits to complete..."
wait

# Count successes and failures
for pid in "${PIDS[@]}"; do
    if wait $pid; then
        ((FIXED_SUCCESS++))
    else
        ((FIXED_FAILED++))
    fi
done

print_success "PHASE 1 COMPLETE: $FIXED_SUCCESS succeeded, $FIXED_FAILED failed"

# Clear PID array
PIDS=()

################################################################################
# PHASE 2: RANDOM SPLITS
################################################################################

print_header "PHASE 2: RUNNING RANDOM SPLITS (${#DATASETS[@]} datasets)"

RANDOM_SUCCESS=0
RANDOM_FAILED=0

for dataset in "${DATASETS[@]}"; do
    wait_for_slot  # Wait if max parallel jobs reached
    
    # Run in background
    (
        if run_experiment "$dataset" "random"; then
            exit 0
        else
            exit 1
        fi
    ) &
    
    # Store PID for tracking
    PIDS+=($!)
    
    sleep 2  # Small delay between launches
done

# Wait for all random splits to complete
print_info "Waiting for all RANDOM splits to complete..."
wait

# Count successes and failures
for pid in "${PIDS[@]}"; do
    if wait $pid; then
        ((RANDOM_SUCCESS++))
    else
        ((RANDOM_FAILED++))
    fi
done

print_success "PHASE 2 COMPLETE: $RANDOM_SUCCESS succeeded, $RANDOM_FAILED failed"

################################################################################
# Summary
################################################################################

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

print_header "FINAL SUMMARY"

echo "Phase 1 (Fixed):  $FIXED_SUCCESS succeeded, $FIXED_FAILED failed"
echo "Phase 2 (Random): $RANDOM_SUCCESS succeeded, $RANDOM_FAILED failed"
echo ""
echo "Total successful: $((FIXED_SUCCESS + RANDOM_SUCCESS)) / $((${#DATASETS[@]} * 2))"
echo "Total failed:     $((FIXED_FAILED + RANDOM_FAILED)) / $((${#DATASETS[@]} * 2))"
echo ""
echo "Time elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Logs saved to: $LOG_DIR/"

# List any failed experiments
TOTAL_FAILED=$((FIXED_FAILED + RANDOM_FAILED))
if [ $TOTAL_FAILED -gt 0 ]; then
    print_warning "Some experiments failed. Check logs in $LOG_DIR/"
    print_info "Failed logs can be identified by searching for 'ERROR' or checking exit codes"
else
    print_success "All experiments completed successfully!"
    print_info "Run analysis script:"
    print_info "  python analyze_investigation5.py"
fi

print_header "COMPLETE"

exit $TOTAL_FAILED
