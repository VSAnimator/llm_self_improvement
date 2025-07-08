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
    --resume)
      RESUME_MODE=true
      RESUME_SEGMENT="$2"
      shift 2
      ;;
    --resume_tasks)
      # Get task numbers for each trial
      RESUME_TASK_1="$2"
      RESUME_TASK_2="$3"
      RESUME_TASK_3="$4"
      RESUME_TASK_4="$5"
      RESUME_TASK_5="$6"
      shift 6
      ;;
    --help)
      echo "Usage: $0 --config CONFIG_STRING [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --config CONFIG_STRING    Configuration string (format: base:env:agent:llm:custom)"
      echo "  --resume SEGMENT          Resume from specific segment number"
      echo "  --resume_tasks T1 T2 T3 T4 T5  Task numbers for each trial when resuming"
      echo "  --help                    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

PBT_RUN_LABEL="${NUM_IC}ic_seg${INITIAL_SEGMENT_SIZE}"  # Auto-generated label

# Define the growth factor for segment size (e.g., 2.0 for doubling)
SEGMENT_GROWTH_FACTOR=${SEGMENT_GROWTH_FACTOR:-2.0}

# Create a single log file for the entire run
LOG_FILE="$CURRENT_DIR/pbt_${ENV_TYPE}_${PBT_RUN_LABEL}_log.txt"
touch "$LOG_FILE"

# Log function to write to the central log file
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting PBT run for environment: ${ENV_TYPE}"
log "Using initial folder: ${SOURCE_DATA_PATH}"
log "Number of in-context examples: ${NUM_IC}"
log "Initial segment size: ${INITIAL_SEGMENT_SIZE}"
log "Run label: ${PBT_RUN_LABEL}"
log "Segment growth factor: ${SEGMENT_GROWTH_FACTOR}"

# Function to clean up background processes on script exit
cleanup() {
    log "Terminating processes..."
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

# Create base directories for trials if they don't exist
for trial in {1..5}; do
    # Copy the initial folder to create each trial's starting folder
    if [ ! -d "$CURRENT_DIR/data/${ENV_TYPE}_pbt_${PBT_RUN_LABEL}_trial_${trial}" ]; then
        cp -r "$CURRENT_DIR/data/${SOURCE_DATA_PATH}" "$CURRENT_DIR/data/${ENV_TYPE}_pbt_${PBT_RUN_LABEL}_trial_${trial}"
        log "Created initial directory for trial $trial"
    fi
done

# Check if we're resuming from specific task numbers
if [[ "$RESUME_MODE" == "true" ]]; then
    log "Resuming from specified task numbers"
    
    if [ -z "$RESUME_SEGMENT" ]; then
        echo "Error: Must specify segment number when resuming"
        exit 1
    fi
    
    # Initialize current task counters with resume values
    declare -a current_tasks=($RESUME_TASK_1 $RESUME_TASK_2 $RESUME_TASK_3 $RESUME_TASK_4 $RESUME_TASK_5)
    
    # Set segment to the resume segment
    segment=$RESUME_SEGMENT
    
    # Calculate segment size based on the segment number
    current_segment_size=$INITIAL_SEGMENT_SIZE
    for ((i=1; i<segment; i++)); do
        current_segment_size=$(python -c "import math; print(int(math.ceil($current_segment_size * $SEGMENT_GROWTH_FACTOR)))")
    done
    
    log "Resuming at segment $segment with segment size $current_segment_size"
    log "Resume task numbers: ${current_tasks[*]}"
else
    # Initialize current task counters for each trial (all starting at 0)
    declare -a current_tasks=(0 0 0 0 0)
    
    # Calculate number of segments dynamically based on growth factor
    segment=1
    remaining_tasks=$NUM_TASKS
    current_segment_size=$INITIAL_SEGMENT_SIZE
    NUM_SEGMENTS=0
    
    while [ $remaining_tasks -gt 0 ]; do
        remaining_tasks=$((remaining_tasks - current_segment_size))
        current_segment_size=$(python -c "import math; print(int(math.ceil($current_segment_size * $SEGMENT_GROWTH_FACTOR)))")
        NUM_SEGMENTS=$((NUM_SEGMENTS + 1))
        
        # Cap the segment size to remaining tasks
        if [ $current_segment_size -gt $remaining_tasks ] && [ $remaining_tasks -gt 0 ]; then
            current_segment_size=$remaining_tasks
        fi
    done
    
    log "Calculated $NUM_SEGMENTS segments with growth factor $SEGMENT_GROWTH_FACTOR"
    
    # Reset for actual run
    segment=1
    current_segment_size=$INITIAL_SEGMENT_SIZE
fi

# Calculate NUM_SEGMENTS if resuming
if [[ "$RESUME_MODE" == "true" ]]; then
    temp_segment=1
    temp_segment_size=$INITIAL_SEGMENT_SIZE
    remaining_tasks=$NUM_TASKS
    NUM_SEGMENTS=0
    
    while [ $remaining_tasks -gt 0 ]; do
        remaining_tasks=$((remaining_tasks - temp_segment_size))
        temp_segment_size=$(python -c "import math; print(int(math.ceil($temp_segment_size * $SEGMENT_GROWTH_FACTOR)))")
        NUM_SEGMENTS=$((NUM_SEGMENTS + 1))
        
        # Cap the segment size to remaining tasks
        if [ $temp_segment_size -gt $remaining_tasks ] && [ $remaining_tasks -gt 0 ]; then
            temp_segment_size=$remaining_tasks
        fi
    done
    
    log "Total segments: $NUM_SEGMENTS"
fi

# Main loop for population-based training
while [ $segment -le $NUM_SEGMENTS ]; do
    log "Starting segment $segment of $NUM_SEGMENTS with segment size $current_segment_size"
    
    PIDS=()
    
    # Run each trial for the current segment
    for trial in {1..5}; do
        current_task=${current_tasks[$((trial-1))]}
        
        # Skip if we've already reached the end
        if [ $current_task -ge $NUM_TASKS ]; then
            log "Trial $trial has completed all tasks, skipping"
            continue
        fi
        
        # Calculate end task for this segment
        # When resuming, we need to ensure we're using the correct segment size
        # for the current trial's progress
        if [[ "$RESUME_MODE" == "true" ]]; then
            # Calculate the target end task based on segment boundaries
            target_task=0
            temp_segment=1
            temp_segment_size=$INITIAL_SEGMENT_SIZE
            
            while [ $temp_segment -le $segment ]; do
                target_task=$((target_task + temp_segment_size))
                
                if [ $temp_segment -lt $segment ]; then
                    temp_segment_size=$(python -c "import math; print(int(math.ceil($temp_segment_size * $SEGMENT_GROWTH_FACTOR)))")
                fi
                
                temp_segment=$((temp_segment + 1))
            done
            
            end_task=$target_task
        else
            # Normal calculation for non-resume mode
            end_task=$((current_task + current_segment_size))
        fi
        
        # Cap at total tasks regardless of mode
        if [ $end_task -gt $NUM_TASKS ]; then
            end_task=$NUM_TASKS
        fi
        
        log "Running trial $trial from task $current_task to $end_task"
        
        # Echo the training script command for this segment
        echo "Would run: python scripts/run_agent.py \
            --llm ${LLM:-openai/gpt-4o-mini} \
            --agent_type ${AGENT_TYPE} \
            --db_path \"$CURRENT_DIR/data/${ENV_TYPE}_pbt_${PBT_RUN_LABEL}_trial_${trial}/learning.db\" \
            --store_episodes \
            --env ${ENV_TYPE} \
            --log_name pbt_${CONFIG_PARTS[0]}_${CONFIG_PARTS[1]}_${CONFIG_PARTS[2]}_${CONFIG_PARTS[3]}_${CONFIG_PARTS[4]}_trial_${trial}_segment_${segment} \
            --num_ic ${NUM_IC} \
            --num_tasks $end_task \
            --num_passes 1 \
            --split train \
            --start_task $current_task"
        
        # Capture process ID and add to array
        PIDS+=($!)
        
        log "Trial $trial segment $segment started with PID ${PIDS[-1]}"
        
        # Update current task for next segment
        current_tasks[$((trial-1))]=$end_task
        
        sleep 5
    done
    
    # Wait for all trials in this segment to complete
    for pid in "${PIDS[@]}"; do
        wait $pid
    done
    
    log "Segment $segment completed for all trials"
    
    # Skip population-based selection if this is the last segment
    if [ $segment -eq $NUM_SEGMENTS ]; then
        log "Final segment completed, skipping population-based selection"
        break
    fi

    # Checkpoint all the DB folders into a specific backups folder
    for trial in {1..5}; do
        # Create a dedicated backups directory
        mkdir -p "$CURRENT_DIR/data/${ENV_TYPE}_pbt_${PBT_RUN_LABEL}_backups/trial_${trial}/segment_${segment}"
        
        # Copy the trial data to the backups folder
        cp -r "$CURRENT_DIR/data/${ENV_TYPE}_pbt_${PBT_RUN_LABEL}_trial_${trial}/"* \
            "$CURRENT_DIR/data/${ENV_TYPE}_pbt_${PBT_RUN_LABEL}_backups/trial_${trial}/segment_${segment}/"
        
        # Also create the checkpoint in the original location for backward compatibility
        #mkdir -p "$CURRENT_DIR/data/${ENV_TYPE}_pbt_${PBT_RUN_LABEL}_trial_${trial}/checkpoint.segment_${segment}"
        #cp -r "$CURRENT_DIR/data/${ENV_TYPE}_pbt_${PBT_RUN_LABEL}_trial_${trial}/"* \
        #    "$CURRENT_DIR/data/${ENV_TYPE}_pbt_${PBT_RUN_LABEL}_trial_${trial}/checkpoint.segment_${segment}/"
        
        log "Created checkpoint for trial $trial segment $segment"
    done
    
    # Evaluate performance on the last segment
    declare -a accuracies=()
    
    for trial in {1..5}; do
        # Calculate accuracy for this trial on the last segment
        log "Calculating accuracy for trial $trial on segment $segment"
        
        # In a real implementation, you would calculate this from results
        accuracy=$(python -c "from scripts.folder_acc import calculate_accuracy; print(calculate_accuracy('logs/episodes/${ENV_TYPE}/train/${AGENT_TYPE}/${LLM:-openai/gpt-4o-mini}/pbt_${PBT_RUN_LABEL}_trial_${trial}_segment_${segment}', segment_size=$current_segment_size))")
        accuracies+=($accuracy)
        
        log "Trial $trial accuracy: $accuracy"
    done
    
    # Find best and worst trials
    best_trial=1
    worst_trial=1
    best_accuracy=${accuracies[0]}
    worst_accuracy=${accuracies[0]}
    
    for trial in {2..5}; do
        idx=$((trial-1))
        if (( $(echo "${accuracies[$idx]} > $best_accuracy" | bc -l) )); then
            best_accuracy=${accuracies[$idx]}
            best_trial=$trial
        fi
        
        if (( $(echo "${accuracies[$idx]} < $worst_accuracy" | bc -l) )); then
            worst_accuracy=${accuracies[$idx]}
            worst_trial=$trial
        fi
    done
    
    log "Best trial: $best_trial with accuracy $best_accuracy"
    log "Worst trial: $worst_trial with accuracy $worst_accuracy"
    
    # Replace worst trial with a copy of the best trial
    if [ $best_trial -ne $worst_trial ]; then
        log "Replacing trial $worst_trial with a copy of trial $best_trial"

        # Log the replacement made
        echo "Replacing trial $worst_trial with a copy of trial $best_trial" >> "$CURRENT_DIR/data/${ENV_TYPE}_pbt_${PBT_RUN_LABEL}_trial_${worst_trial}/replacement_log.txt"

        # Copy the best trial's folder to the worst trial
        rm -rf "$CURRENT_DIR/data/${ENV_TYPE}_pbt_${PBT_RUN_LABEL}_trial_${worst_trial}"
        cp -r "$CURRENT_DIR/data/${ENV_TYPE}_pbt_${PBT_RUN_LABEL}_trial_${best_trial}" \
             "$CURRENT_DIR/data/${ENV_TYPE}_pbt_${PBT_RUN_LABEL}_trial_${worst_trial}"

        # Reset the current task for the worst trial to match the best trial
        current_tasks[$((worst_trial-1))]=${current_tasks[$((best_trial-1))]}
    else
        log "Best and worst trials are the same, no replacement needed"
    fi
    
    log "Population-based selection completed for segment $segment"
    
    # Calculate next segment size using growth factor
    current_segment_size=$(python -c "import math; print(int(math.ceil($current_segment_size * $SEGMENT_GROWTH_FACTOR)))")
    log "Next segment size will be $current_segment_size"
    
    # Increment segment counter
    segment=$((segment + 1))
done

log "All segments completed"

# Create a directory for the archive if it doesn't exist
ARCHIVE_DIR="$CURRENT_DIR/data/${ENV_TYPE}_archives"
mkdir -p "$ARCHIVE_DIR"

# Get current date and time for the archive name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_NAME="${ENV_TYPE}_pbt_${PBT_RUN_LABEL}_trials_${TIMESTAMP}.tar.gz"

log "Creating compressed archive of PBT databases..."
