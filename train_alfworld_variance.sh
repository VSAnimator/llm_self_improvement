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

# Create directories for 5 trials
for trial in {1..5}; do
    echo "Creating directory for trial $trial"
    cp -r "$CURRENT_DIR/data/alfworld_expel" "$CURRENT_DIR/data/alfworld_expel_trial_${trial}_6_ic"
done

# Wait for all copy operations to complete
wait

PIDS=()

# Run 5 trials in parallel
for trial in {1..5}; do
    echo "Starting trial $trial"
    
    # Run the training script in the background
    python scripts/run_agent_v2.py \
        --llm openai/gpt-4o-mini \
        --agent_type rap_flex \
        --db_path "$CURRENT_DIR/data/alfworld_expel_trial_${trial}_6_ic/learning.db" \
        --store_episodes \
        --env alfworld \
        --log_name trial_${trial}_6_ic \
        --num_ic 6 \
        --num_tasks 3500 \
        --num_passes 1 &
    
    # Capture process ID and add to array
    PIDS+=($!)
    
    echo "Trial $trial started with PID ${PIDS[-1]}"
done

# Wait for all trials to complete
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "All trials completed"



