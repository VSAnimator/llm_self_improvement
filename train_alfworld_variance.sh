# Get current directory
CURRENT_DIR=$(pwd)

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

'''
# Create directories for 5 trials
for trial in {1..5}; do
    echo "Creating directory for trial $trial"
    cp -r "$CURRENT_DIR/data/alfworld_expel" "$CURRENT_DIR/data/alfworld_expel_trial_${trial}_6_ic"
done

# Wait for all copy operations to complete
wait
'''

PIDS=()

# Define arrays for trial start tasks
trial_start_tasks_3=(1384 1175 1792 1588 1377)
trial_start_tasks_6=(1712 3076 3500 2236 3500)

# Run 5 trials in parallel
for trial in {1..5}; do
    echo "Starting trial $trial"

    # Use proper bash array syntax with index
    trial_start_task=${trial_start_tasks_3[$((trial-1))]}

    echo "Trial start task: $trial_start_task"
    
    # If the start task is 3500, then skip the trial
    if [ $trial_start_task -eq 3500 ]; then
        echo "Skipping trial $trial because start task is 3500"
        continue
    fi
    
    # Run the training script in the background
    python scripts/run_agent_v2.py \
        --llm openai/gpt-4o-mini \
        --agent_type rap_flex \
        --db_path "$CURRENT_DIR/data/alfworld_expel_trial_${trial}/learning.db" \
        --store_episodes \
        --env alfworld \
        --log_name trial_${trial} \
        --num_ic 3 \
        --num_tasks 3500 \
        --num_passes 1 \
        --start_task $trial_start_task &
    
    # Capture process ID and add to array
    PIDS+=($!)
    
    echo "Trial $trial started with PID ${PIDS[-1]}"

    sleep 5
done

# Wait for all trials to complete
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "All trials completed"

# Create a directory for the archive if it doesn't exist
ARCHIVE_DIR="$CURRENT_DIR/data/alfworld_archives"
mkdir -p "$ARCHIVE_DIR"

# Get current date and time for the archive name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_NAME="alfworld_expel_trials_${TIMESTAMP}.tar.gz"

echo "Creating compressed archive of specific backup databases..."




