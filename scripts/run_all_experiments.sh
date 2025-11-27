#!/bin/bash

################################################################################
# Parallel Experiment Runner for SGC Anti-Smoothing Part A/B Framework
################################################################################

# Configuration
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
K_VALUES="2 4 6 8 10"
MAX_PARALLEL_JOBS=3  # Run 3 experiments in parallel 

# Directories
SCRIPT_PATH="experiments/investigation_sgc_antiSmoothing_partAB.py"
LOG_DIR="logs/parallel_experiments"
RESULTS_DIR="results/investigation_sgc_antiSmoothing"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Functions
################################################################################

print_header() {
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}→ $1${NC}"
}

# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local split_type=$2
    local log_file="$LOG_DIR/${dataset}_${split_type}.log"
    
    print_info "Starting: $dataset ($split_type)"
    
    # Run experiment and capture output
    python "$SCRIPT_PATH" "$dataset" \
        --k_diffusion $K_VALUES \
        --splits "$split_type" \
        --component lcc \
        > "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "COMPLETED: $dataset ($split_type)"
        echo "COMPLETED" > "$LOG_DIR/${dataset}_${split_type}.status"
    else
        print_error "FAILED: $dataset ($split_type) - Check $log_file"
        echo "FAILED" > "$LOG_DIR/${dataset}_${split_type}.status"
    fi
    
    return $exit_code
}

# Function to check if experiment is already complete
is_complete() {
    local dataset=$1
    local split_type=$2
    local result_file="$RESULTS_DIR/${dataset}_${split_type}_lcc/k10/metrics/results.json"
    
    if [ -f "$result_file" ]; then
        return 0  # Already complete
    else
        return 1  # Not complete
    fi
}

# Function to count running jobs
count_running_jobs() {
    jobs -r | wc -l
}

# Function to wait for available slot
wait_for_slot() {
    while [ $(count_running_jobs) -ge $MAX_PARALLEL_JOBS ]; do
        sleep 5
    done
}

################################################################################
# Main Execution
################################################################################

print_header "SGC ANTI-SMOOTHING EXPERIMENTS - PARALLEL EXECUTION"

echo ""
echo "Configuration:"
echo "  Datasets: ${#DATASETS[@]}"
echo "  Split types: ${#SPLIT_TYPES[@]}"
echo "  Total experiments: $((${#DATASETS[@]} * ${#SPLIT_TYPES[@]}))"
echo "  K values: $K_VALUES"
echo "  Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "  Log directory: $LOG_DIR"
echo ""

# Clear old status files
rm -f "$LOG_DIR"/*.status

# Start time
START_TIME=$(date +%s)

# Counter for experiments
TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#SPLIT_TYPES[@]}))
SKIPPED=0
QUEUED=0

print_header "CHECKING EXISTING RESULTS & QUEUEING EXPERIMENTS"

# Queue all experiments
for dataset in "${DATASETS[@]}"; do
    for split_type in "${SPLIT_TYPES[@]}"; do
        if is_complete "$dataset" "$split_type"; then
            print_warning "SKIP: $dataset ($split_type) - already complete"
            ((SKIPPED++))
        else
            print_info "QUEUE: $dataset ($split_type)"
            ((QUEUED++))
            
            # Wait for available slot
            wait_for_slot
            
            # Run in background
            run_experiment "$dataset" "$split_type" &
            
            # Small delay to avoid race conditions
            sleep 2
        fi
    done
done

echo ""
print_header "WAITING FOR ALL EXPERIMENTS TO COMPLETE"
echo ""
echo "Queued: $QUEUED experiments"
echo "Skipped: $SKIPPED experiments (already complete)"
echo ""

# Wait for all background jobs to finish
wait

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

################################################################################
# Final Summary
################################################################################

print_header "EXPERIMENT SUMMARY"

echo ""
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# Count results
COMPLETED=0
FAILED=0

for dataset in "${DATASETS[@]}"; do
    for split_type in "${SPLIT_TYPES[@]}"; do
        status_file="$LOG_DIR/${dataset}_${split_type}.status"
        if [ -f "$status_file" ]; then
            status=$(cat "$status_file")
            if [ "$status" = "COMPLETED" ]; then
                ((COMPLETED++))
            elif [ "$status" = "FAILED" ]; then
                ((FAILED++))
            fi
        fi
    done
done

echo "Results:"
echo "  ✓ Completed: $COMPLETED"
echo "  ✗ Failed: $FAILED"
echo "  ⊘ Skipped: $SKIPPED (already done)"
echo ""

if [ $FAILED -gt 0 ]; then
    print_warning "FAILED EXPERIMENTS:"
    echo ""
    for dataset in "${DATASETS[@]}"; do
        for split_type in "${SPLIT_TYPES[@]}"; do
            status_file="$LOG_DIR/${dataset}_${split_type}.status"
            if [ -f "$status_file" ] && [ "$(cat $status_file)" = "FAILED" ]; then
                log_file="$LOG_DIR/${dataset}_${split_type}.log"
                print_error "  $dataset ($split_type) - see $log_file"
            fi
        done
    done
    echo ""
fi

if [ $COMPLETED -gt 0 ]; then
    print_success "Successfully completed $COMPLETED experiments!"
    echo ""
    echo "Next steps:"
    echo "  1. Run analysis: python scripts/analyze_partAB.py --split_type fixed"
    echo "  2. Run analysis: python scripts/analyze_partAB.py --split_type random"
    echo "  3. Generate report: python scripts/analyze_antiSmoothing_complete.py"
fi

print_header "ALL DONE"
