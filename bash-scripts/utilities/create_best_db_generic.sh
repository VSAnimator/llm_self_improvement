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

# Set environment variables (can be overridden before running the script)
# Parse command line arguments
if [ "$#" -lt 3 ]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 <SOURCE_DIR> <TARGET_DIR_BASE> <ENV_NAME> <JSON_SUFFIX>"
    echo "  SOURCE_DIR: Source template directory"
    echo "  TARGET_DIR_BASE: Base directory for filtered data"
    echo "  ENV_NAME: Environment name"
    echo "  JSON_SUFFIX: JSON file suffix"
    exit 1
fi

#SOURCE_DIR="$1"
TARGET_DIR_BASE="$1"
ENV_NAME="$2"
JSON_SUFFIX="$3"

# Create directories for best and worst examples
#for mode in best worst; do
#    echo "Creating directory for $mode examples"
#    cp -r "$SOURCE_DIR" "$TARGET_DIR_BASE/${ENV_NAME}_${mode}_examples"
#done

# Ingest the best and worst examples into the databases
for mode in best worst; do
    echo "Ingesting $mode examples"
    python src/llm_agent/database/ingest_alfworld_db_entries.py --mode $mode --json_file "compare_${JSON_SUFFIX}/${mode}_examples_per_task.json" --db_path "$TARGET_DIR_BASE/${ENV_NAME}_${mode}_examples/learning.db"
done
