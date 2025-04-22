#!/bin/bash

# Get current directory
CURRENT_DIR=$(pwd)

# Source configuration loader with default values
source "$CURRENT_DIR/agent_configs/load_config.sh" --base default

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      # Parse config string (format: "base:env:agent:llm:custom")
      IFS=':' read -ra CONFIG_PARTS <<< "$2"
      if [[ -n "${CONFIG_PARTS[0]}" ]]; then source "$CURRENT_DIR/agent_configs/base/${CONFIG_PARTS[0]}.sh" 2>/dev/null; fi
      if [[ -n "${CONFIG_PARTS[1]}" ]]; then source "$CURRENT_DIR/agent_configs/env/${CONFIG_PARTS[1]}.sh" 2>/dev/null; fi
      if [[ -n "${CONFIG_PARTS[2]}" ]]; then source "$CURRENT_DIR/agent_configs/agent/${CONFIG_PARTS[2]}.sh" 2>/dev/null; fi
      if [[ -n "${CONFIG_PARTS[3]}" ]]; then source "$CURRENT_DIR/agent_configs/llm/${CONFIG_PARTS[3]}.sh" 2>/dev/null; fi
      if [[ -n "${CONFIG_PARTS[4]}" ]]; then source "$CURRENT_DIR/agent_configs/custom/${CONFIG_PARTS[4]}.sh" 2>/dev/null; fi
      shift 2
      ;;
    # Keep all existing command line options for overriding config values
    --env)
      ENV_TYPE="$2"
      shift 2
      ;;
    --agent_type)
      AGENT_TYPE="$2"
      shift 2
      ;;
    --llm)
      LLM="$2"
      shift 2
      ;;
    --num_tasks)
      NUM_TASKS="$2"
      shift 2
      ;;
    --start_task)
      START_TASK="$2"
      shift 2
      ;;
    --num_ic)
      NUM_IC="$2"
      shift 2
      ;;
    --num_trials)
      NUM_TRIALS="$2"
      shift 2
      ;;
    --log_name)
      LOG_NAME="$2"
      shift 2
      ;;
    --no_copy_data)
      COPY_DATA=false
      shift
      ;;
    --source_data_path)
      SOURCE_DATA_PATH="$2"
      shift 2
      ;;
    --parallel)
      PARALLEL="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --config CONFIG_STRING    Configuration string (format: base:env:agent:llm:custom)"
      echo "  --env ENV_TYPE            Environment type (alfworld, intercode_sql, wordcraft)"
      echo "  --agent_type AGENT_TYPE   Agent type (rap_flex, rap_noplan, etc.)"
      echo "  --llm LLM                 LLM model to use"
      echo "  --num_tasks NUM_TASKS     Number of tasks to run"
      echo "  --start_task START_TASK   Task to start from (default: 0, use non-zero for resuming)"
      echo "  --num_ic NUM_IC           Number of in-context examples"
      echo "  --num_trials NUM_TRIALS   Number of trials to run"
      echo "  --log_name LOG_NAME       Custom directory name for logs"
      echo "  --no_copy_data            Don't copy data directory (for resuming runs)"
      echo "  --source_data_path PATH   Source data path to copy from"
      echo "  --parallel NUM            Number of parallel runs"
      echo "  --help                    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Function to clean up background processes on script exit
cleanup() {
    echo "Terminating processes..."
    for pid in "${PIDS[@]}"; do
        if [[ -n "$pid" ]]; then
            kill $pid 2>/dev/null
            wait $pid 2>/dev/null
        fi
    done
    exit 1
}

# Trap Ctrl+C and call cleanup function
trap cleanup SIGINT

PIDS=()

# Function to run agent for a specific trial
run_agent() {
    local env=$1
    local trial=$2
    local trial_db_path=$3
    local trial_log_name=$4
    local start_task=$5
    local sleep_duration=${6:-5}
    
    echo "Starting trial $trial for $env"
    
    # Run the training script in the background
    python scripts/run_agent_v2.py \
        --llm "$LLM" \
        --agent_type "$AGENT_TYPE" \
        --num_passes 1 \
        --env "$env" \
        --num_ic "$NUM_IC" \
        --num_tasks "$NUM_TASKS" \
        --start_task "$start_task" \
        --db_path "$trial_db_path" \
        --log_name "$trial_log_name" \
        --store_episodes &
    
    # Capture process ID and add to array
    PIDS+=($!)
    
    echo "Trial $trial started with PID ${PIDS[-1]}"
    
    sleep "$sleep_duration"
}

# Create a counter for parallel jobs
JOB_COUNT=0

# Run trials for the selected environment
for trial in $(seq 1 $NUM_TRIALS); do
  # Create data directory for trial if needed
  trial_data_dir="$CURRENT_DIR/data/${ENV_TYPE}/trial_${trial}"
  
  if [ "$COPY_DATA" = true ]; then
    echo "Creating directory for trial $trial"
    if [ -n "$SOURCE_DATA_PATH" ]; then
      cp -r "$SOURCE_DATA_PATH" "$trial_data_dir"
    else
      mkdir -p "$trial_data_dir"
    fi
  fi
  
  # Set trial DB path
  TRIAL_DB_PATH="${trial_data_dir}/learning.db"
  
  # Set log name
  if [ -z "$LOG_NAME" ]; then
    TRIAL_LOG_NAME="${ENV_TYPE}_trial_${trial}"
  else
    TRIAL_LOG_NAME="${LOG_NAME}_trial_${trial}"
  fi
  
  # Run agent with appropriate parameters
  run_agent "$ENV_TYPE" "$trial" "$TRIAL_DB_PATH" "$TRIAL_LOG_NAME" "$START_TASK" "5"
  
  # Track parallel jobs
  ((JOB_COUNT++))
  
  # Wait if we've reached the parallel limit
  if [ "$JOB_COUNT" -ge "$PARALLEL" ]; then
    wait "${PIDS[-$PARALLEL]}"
    JOB_COUNT=$((PARALLEL-1))
  fi
done

# Wait for all trials to complete
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo "All trials completed"