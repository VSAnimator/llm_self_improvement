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
# Create directories for best and worst examples
for mode in best worst; do
    for ic in "_3_ic" "_6_ic"; do
        echo "Creating directory for $mode examples with $ic"
        cp -r "$CURRENT_DIR/data/alfworld_expel" "$CURRENT_DIR/data/alfworld_filtered/alfworld_${mode}_examples${ic}"
    done
done
'''

# Ingest the best and worst examples into the databases
for mode in best worst; do
    for ic in "_3_ic" "_6_ic"; do
        echo "Ingesting $mode examples with $ic"
        python src/llm_agent/database/ingest_alfworld_db_entries.py --mode $mode --json_file "compare${ic}/${mode}_examples_per_task.json" --db_path "data/alfworld_filtered/alfworld_${mode}_examples${ic}/learning.db"
    done
done


'''
# Ingest the best and worst examples into the databases
for mode in best worst; do
    for ic in "_3_ic" "_6_ic"; do
        echo "Ingesting $mode examples with $ic"
        python src/llm_agent/database/ingest_alfworld_logs.py --mode $mode --json_file "compare${ic}/${mode}_examples_per_task.json" --log_file_base "logs/episodes/alfworld/train/rap_flex/openai/gpt-4o-mini/" --db_path "data/alfworld_filtered/alfworld_${mode}_examples${ic}/learning.db" --task_offset 18
    done
done
'''