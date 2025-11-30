#!/bin/bash

################################################################################
# Investigation 4: Run all datasets with parallel execution
################################################################################

# Configuration
MAX_PARALLEL=2
K_DIFFUSION=10
COMPONENT_TYPE="lcc"

# Dataset list
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

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local split_type=$2
    
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} Starting: ${dataset} (${split_type})"
    
    # Create log directory
    mkdir -p logs/investigation4
    
    # Log file
    local log_file="logs/investigation4/${dataset}_${split_type}_k${K_DIFFUSION}.log"
    
    # Run experiment
    python experiments/investigation4_spectral_normalization.py \
        --dataset ${dataset} \
        --k_diffusion ${K_DIFFUSION} \
        --split_type ${split_type} \
        --component_type ${COMPONENT_TYPE} \
        > ${log_file} 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} ✓ Completed: ${dataset} (${split_type})"
    else
        echo -e "${RED}[$(date '+%H:%M:%S')]${NC} ✗ Failed: ${dataset} (${split_type}) - Check ${log_file}"
    fi
    
    return $exit_code
}

# Function to run experiments in parallel batches
run_parallel_batch() {
    local split_type=$1
    local total=${#DATASETS[@]}
    local completed=0
    
    echo ""
    echo -e "${YELLOW}======================================================================${NC}"
    echo -e "${YELLOW}Running ${split_type^^} split experiments (${total} datasets, ${MAX_PARALLEL} parallel)${NC}"
    echo -e "${YELLOW}======================================================================${NC}"
    echo ""
    
    # Array to track background processes
    local pids=()
    local running=0
    
    for dataset in "${DATASETS[@]}"; do
        # Wait if we've reached max parallel jobs
        while [ $running -ge $MAX_PARALLEL ]; do
            # Check which processes have finished
            for i in "${!pids[@]}"; do
                if ! kill -0 ${pids[$i]} 2>/dev/null; then
                    # Process finished
                    wait ${pids[$i]}
                    unset pids[$i]
                    ((running--))
                    ((completed++))
                    echo -e "${BLUE}Progress: ${completed}/${total} completed${NC}"
                fi
            done
            sleep 1
        done
        
        # Start new experiment in background
        run_experiment ${dataset} ${split_type} &
        local pid=$!
        pids+=($pid)
        ((running++))
        
        # Small delay to avoid overwhelming the system
        sleep 2
    done
    
    # Wait for remaining jobs to complete
    echo ""
    echo -e "${BLUE}Waiting for remaining experiments to complete...${NC}"
    for pid in "${pids[@]}"; do
        wait $pid
        ((completed++))
        echo -e "${BLUE}Progress: ${completed}/${total} completed${NC}"
    done
    
    echo ""
    echo -e "${GREEN}✓ All ${split_type} split experiments completed!${NC}"
    echo ""
}

################################################################################
# Main execution
################################################################################

echo -e "${YELLOW}======================================================================${NC}"
echo -e "${YELLOW}INVESTIGATION 4: SPECTRAL NORMALIZATION - ALL DATASETS${NC}"
echo -e "${YELLOW}======================================================================${NC}"
echo ""
echo "Configuration:"
echo "  Datasets: ${#DATASETS[@]}"
echo "  k_diffusion: ${K_DIFFUSION}"
echo "  Component type: ${COMPONENT_TYPE}"
echo "  Max parallel: ${MAX_PARALLEL}"
echo "  Total experiments: $((${#DATASETS[@]} * 2)) (${#DATASETS[@]} fixed + ${#DATASETS[@]} random)"
echo ""
echo "Datasets to process:"
for dataset in "${DATASETS[@]}"; do
    echo "  - ${dataset}"
done
echo ""

read -p "Press Enter to start, or Ctrl+C to cancel..."

# Create logs directory
mkdir -p logs/investigation4

# Record start time
START_TIME=$(date +%s)

# Run all FIXED split experiments first
run_parallel_batch "fixed"

# Run all RANDOM split experiments second
run_parallel_batch "random"

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}ALL EXPERIMENTS COMPLETED!${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results location: results/investigation4_spectral_normalization/"
echo "Logs location: logs/investigation4/"
echo ""
echo "Next step: Run analysis script"
echo "  python scripts/analyze_investigation4.py --split_type fixed"
echo "  python scripts/analyze_investigation4.py --split_type random"
echo ""
