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

# Ingest the best and worst examples into the databases (other mode is _3_ic)
for mode in best worst; do
    for ic in "_6_ic"; do 
        echo "Ingesting $mode examples with $ic"
        python src/llm_agent/database/ingest_alfworld_db_entries.py --mode $mode --json_file "compare${ic}/${mode}_examples_per_task.json" --db_path "data/alfworld_filtered/alfworld_pbt_${mode}_examples${ic}/learning.db"
    done
done

