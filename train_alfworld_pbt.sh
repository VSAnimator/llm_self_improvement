# Get current directory
CURRENT_DIR=$(pwd)

# Define a label for this PBT run - change this for different runs
PBT_RUN_LABEL="6ic_seg20"

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

# Create base directories for trials if they don't exist
for trial in {1..5}; do
    # Copy the alfworld_expel folder to create each trial's starting folder
    if [ ! -d "$CURRENT_DIR/data/alfworld_pbt_${PBT_RUN_LABEL}_trial_${trial}" ]; then
        cp -r "$CURRENT_DIR/data/alfworld_expel" "$CURRENT_DIR/data/alfworld_pbt_${PBT_RUN_LABEL}_trial_${trial}"
    fi
done

# Define the total number of tasks and segment size
TOTAL_TASKS=500
SEGMENT_SIZE=20
NUM_SEGMENTS=$((TOTAL_TASKS / SEGMENT_SIZE))

# Initialize current task counters for each trial (all starting at 0)
declare -a current_tasks=(0 0 0 0 0)

# Main loop for population-based training
for segment in $(seq 1 $NUM_SEGMENTS); do
    echo "Starting segment $segment of $NUM_SEGMENTS"
    
    PIDS=()
    
    # Run each trial for the current segment
    for trial in {1..5}; do
        current_task=${current_tasks[$((trial-1))]}
        
        # Skip if we've already reached the end
        if [ $current_task -ge $TOTAL_TASKS ]; then
            echo "Trial $trial has completed all tasks, skipping"
            continue
        fi
        
        # Calculate end task for this segment
        end_task=$((current_task + SEGMENT_SIZE))
        if [ $end_task -gt $TOTAL_TASKS ]; then
            end_task=$TOTAL_TASKS
        fi
        
        echo "Running trial $trial from task $current_task to $end_task"
        
        # Run the training script for this segment
        python scripts/run_agent_v2.py \
            --llm openai/gpt-4o-mini \
            --agent_type rap_flex \
            --db_path "$CURRENT_DIR/data/alfworld_pbt_${PBT_RUN_LABEL}_trial_${trial}/learning.db" \
            --store_episodes \
            --env alfworld \
            --log_name pbt_${PBT_RUN_LABEL}_trial_${trial}_segment_${segment} \
            --num_ic 3 \
            --num_tasks $end_task \
            --num_passes 1 \
            --start_task $current_task &
        
        # Capture process ID and add to array
        PIDS+=($!)
        
        echo "Trial $trial segment $segment started with PID ${PIDS[-1]}"
        
        # Update current task for next segment
        current_tasks[$((trial-1))]=$end_task
        
        sleep 5
    done
    
    # Wait for all trials in this segment to complete
    for pid in "${PIDS[@]}"; do
        wait $pid
    done
    
    echo "Segment $segment completed for all trials"
    
    # Skip population-based selection if this is the last segment
    if [ $segment -eq $NUM_SEGMENTS ]; then
        echo "Final segment completed, skipping population-based selection"
        continue
    fi

    # Checkpoint all the DB folders into a specific backups folder
    for trial in {1..5}; do
        # Create a dedicated backups directory
        mkdir -p "$CURRENT_DIR/data/alfworld_pbt_${PBT_RUN_LABEL}_backups/trial_${trial}/segment_${segment}"
        
        # Copy the trial data to the backups folder
        cp -r "$CURRENT_DIR/data/alfworld_pbt_${PBT_RUN_LABEL}_trial_${trial}/"* \
            "$CURRENT_DIR/data/alfworld_pbt_${PBT_RUN_LABEL}_backups/trial_${trial}/segment_${segment}/"
        
        # Also create the checkpoint in the original location for backward compatibility
        mkdir -p "$CURRENT_DIR/data/alfworld_pbt_${PBT_RUN_LABEL}_trial_${trial}/checkpoint.segment_${segment}"
        cp -r "$CURRENT_DIR/data/alfworld_pbt_${PBT_RUN_LABEL}_trial_${trial}/"* \
            "$CURRENT_DIR/data/alfworld_pbt_${PBT_RUN_LABEL}_trial_${trial}/checkpoint.segment_${segment}/"
    done
    
    # Evaluate performance on the last segment
    declare -a accuracies=()
    
    for trial in {1..5}; do
        # Calculate accuracy for this trial on the last segment
        # This is a placeholder - replace with actual accuracy calculation
        # For example, you might parse log files or query the database
        
        echo "Calculating accuracy for trial $trial on segment $segment"
        
        # In a real implementation, you would calculate this from results
        accuracy=$(python -c "from scripts.folder_acc import calculate_accuracy; print(calculate_accuracy('logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/pbt_${PBT_RUN_LABEL}_trial_${trial}_segment_${segment}', segment_size=$SEGMENT_SIZE))")
        accuracies+=($accuracy)
        
        echo "Trial $trial accuracy: $accuracy"
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
    
    echo "Best trial: $best_trial with accuracy $best_accuracy"
    echo "Worst trial: $worst_trial with accuracy $worst_accuracy"
    
    # Replace worst trial with a copy of the best trial
    if [ $best_trial -ne $worst_trial ]; then
        echo "Replacing trial $worst_trial with a copy of trial $best_trial"

        # Log the replacement made
        echo "Replacing trial $worst_trial with a copy of trial $best_trial" >> "$CURRENT_DIR/data/alfworld_pbt_${PBT_RUN_LABEL}_trial_${worst_trial}/replacement_log.txt"

        # Copy the best trial's folder to the worst trial
        rm -rf "$CURRENT_DIR/data/alfworld_pbt_${PBT_RUN_LABEL}_trial_${worst_trial}"
        cp -r "$CURRENT_DIR/data/alfworld_pbt_${PBT_RUN_LABEL}_trial_${best_trial}" \
             "$CURRENT_DIR/data/alfworld_pbt_${PBT_RUN_LABEL}_trial_${worst_trial}"

        # Reset the current task for the worst trial to match the best trial
        current_tasks[$((worst_trial-1))]=${current_tasks[$((best_trial-1))]}
    else
        echo "Best and worst trials are the same, no replacement needed"
    fi
    
    echo "Population-based selection completed for segment $segment"
done

echo "All segments completed"

# Create a directory for the archive if it doesn't exist
ARCHIVE_DIR="$CURRENT_DIR/data/alfworld_archives"
mkdir -p "$ARCHIVE_DIR"

# Get current date and time for the archive name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_NAME="alfworld_pbt_${PBT_RUN_LABEL}_trials_${TIMESTAMP}.tar.gz"

echo "Creating compressed archive of PBT databases..."



