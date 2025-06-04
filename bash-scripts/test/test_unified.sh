#!/bin/bash

# Get current directory
CURRENT_DIR=$(pwd)

# Source configuration loader with default values
source "$CURRENT_DIR/agent_configs/load_config.sh" --base default

# Default test parameters
CHECKPOINT=""
TEST_PARALLEL=1

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
      shift 2
      ;;
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
    --num_ic)
      NUM_IC="$2"
      shift 2
      ;;
    --num_trials)
      NUM_TRIALS="$2"
      shift 2
      ;;
    --test_num_tasks)
      TEST_NUM_TASKS="$2"
      shift 2
      ;;
    --log_name)
      LOG_NAME="$2"
      shift 2
      ;;
    --checkpoints)
      CHECKPOINTS="$2"
      shift 2
      ;;
    --parallel)
      TEST_PARALLEL="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --config CONFIG_STRING    Configuration string (format: base:env:agent:llm)"
      echo "  --env ENV_TYPE            Base environment type (alfworld, intercode_sql, wordcraft)"
      echo "  --agent_type AGENT_TYPE   Agent type (rap_flex, rap_noplan, etc.)"
      echo "  --llm LLM                 LLM model to use"
      echo "  --num_ic NUM_IC           Number of in-context examples"
      echo "  --num_trials NUM_TRIALS   Number of trials to run"
      echo "  --test_num_tasks NUM      Number of test tasks to run"
      echo "  --log_name LOG_NAME       Custom directory name for logs"
      echo "  --checkpoints LIST        Comma-separated list of checkpoints to test (e.g. 100,500,1000)"
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

# If no checkpoints specified, test only the latest
if [ -z "$CHECKPOINTS" ]; then
    CHECKPOINTS="$NUM_TASKS"
fi

# Convert comma-separated list to array
IFS=',' read -ra CHECKPOINT_ARRAY <<< "$CHECKPOINTS"

# Create a counter for parallel jobs
JOB_COUNT=0

# Run tests for specified checkpoints across all trials
for trial in $(seq 1 $NUM_TRIALS); do
    for ckpt in "${CHECKPOINT_ARRAY[@]}"; do
        # Set the database path based on trial and checkpoint
        DB_PATH="$CURRENT_DIR/data/${ENV_TYPE}/trial_${trial}"
        
        # For checkpoints other than final, look for backup directories
        if [ "$ckpt" != "$NUM_TASKS" ]; then
            if [ -d "${DB_PATH}_backups/${ckpt}" ]; then
                DB_PATH="${DB_PATH}_backups/${ckpt}/learning.db"
            else
                echo "Warning: Checkpoint ${ckpt} not found for trial ${trial}, skipping"
                continue
            fi
        else
            DB_PATH="${DB_PATH}/learning.db"
        fi
        
        # Set log name for this test run
        if [ -z "$LOG_NAME" ]; then
            TEST_LOG_NAME="${CONFIG_STRING}_trial_${trial}/ckpt_${ckpt}"
        else
            TEST_LOG_NAME="${LOG_NAME}_trial_${trial}/ckpt_${ckpt}"
        fi
        
        echo "Testing checkpoint ${ckpt} for trial ${trial}"
        
        # Comment out the actual run and just echo the command
        echo Would run: python scripts/run_agent_v2.py \
            --llm $LLM \
            --agent_type $AGENT_TYPE \
            --num_passes 1 \
            --env $ENV_TYPE \
            --num_ic $NUM_IC \
            --num_tasks $TEST_NUM_TASKS \
            --db_path $DB_PATH \
            --log_name $TEST_LOG_NAME \
            --split test \
            --parallel $TEST_PARALLEL
        
        # Capture process ID
        PIDS+=($!)
        
        echo "Test for trial ${trial}, checkpoint ${ckpt} started with PID ${PIDS[-1]}"
        
        # Track parallel jobs
        ((JOB_COUNT++))
        
        # Wait if we've reached the parallel limit
        if [ "$JOB_COUNT" -ge "$TEST_PARALLEL" ]; then
            wait "${PIDS[-$TEST_PARALLEL]}"
            JOB_COUNT=$((TEST_PARALLEL-1))
        fi
        
        sleep 3
    done
    
    # Wait for all jobs for this trial to complete
    if [ "$TEST_PARALLEL" -eq 1 ]; then
        wait
        JOB_COUNT=0
    fi
done

# Wait for any remaining processes
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo "All tests completed" 