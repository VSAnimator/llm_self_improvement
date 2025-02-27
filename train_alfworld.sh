#!/bin/bash

# Get current directory
CURRENT_DIR=$(pwd)

# Function to clean up background processes on script exit
cleanup() {
    echo "Terminating processes..."
    if [[ -n "$CHROMA_PID" ]]; then
        kill $CHROMA_PID 2>/dev/null
        wait $CHROMA_PID 2>/dev/null
    fi
    if [[ -n "$PYTHON_PID" ]]; then
        kill $PYTHON_PID 2>/dev/null
        wait $PYTHON_PID 2>/dev/null
    fi
    exit 1
}

# Trap Ctrl+C and call cleanup function
trap cleanup SIGINT

# Wait for all copy operations to complete
wait

# Run training script for each number of in-context examples sequentially
for parallel in 1 2 4 8 16; do

    cp -r "$CURRENT_DIR/data/alfworld_chroma_base" "$CURRENT_DIR/data/alfworld_chroma_parallel_${parallel}" &
    # Start Chroma server in the background, redirecting output to a log file
    chroma run --path "$CURRENT_DIR/data/alfworld_chroma_parallel_${parallel}" --port 8008 &>>chroma_log.txt &

    # Capture Chroma server process ID
    CHROMA_PID=$!

    # Give Chroma some time to initialize
    sleep 5

    # Run the training script in the background
    python scripts/run_agent_v2.py \
        --llm openai/gpt-4o-mini \
        --agent_type rap_flex \
        --db_type chroma \
        --log_name expel_train_parallel_${parallel}_chroma \
        --num_passes 1 \
        --env alfworld \
        --store_episodes \
        --num_ic 3 \
        --parallel $parallel \
        --num_tasks 3500 &

    # Capture Python process ID
    PYTHON_PID=$!

    # Wait for the Python script to finish
    wait $PYTHON_PID

    # Kill the Chroma server process after training completes
    kill $CHROMA_PID

    # Wait for the Chroma server to terminate
    wait $CHROMA_PID 2>/dev/null
done
