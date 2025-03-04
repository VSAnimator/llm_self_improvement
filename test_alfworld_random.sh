# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
python scripts/run_agent_v2.py \
    --llm openai/gpt-4o-mini \
    --agent_type rap \
    --db_path "$CURRENT_DIR/data/alfworld_expel_rap/learning.db" \
    --log_name expel_random_test \
    --num_passes 10 \
    --env alfworld_test \
    --num_tasks 4 \
    --parallel 4 \
    --random_retrieval
