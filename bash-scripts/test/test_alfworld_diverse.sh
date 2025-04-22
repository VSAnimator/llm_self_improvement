# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
python scripts/run_agent_v2.py \
    --llm openai/gpt-4o-mini \
    --agent_type rap_diversity \
    --db_path "$CURRENT_DIR/data/alfworld_expel_diverse/learning.db" \
    --log_name expel_diverse \
    --num_passes 15 \
    --env alfworld_test \
    --num_ic 10 \
    --store_episodes \
    --diversity_mode \
    --num_tasks 4 \
    --start_task 3 
