# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
python scripts/run_agent_v2.py \
    --llm openai/gpt-4o-mini \
    --agent_type rap_refine \
    --db_path "$CURRENT_DIR/data/alfworld_expel_refine/learning.db" \
    --log_name expel_refine \
    --num_passes 5 \
    --env alfworld_test \
    --num_ic 10 \
    --store_episodes \
    --num_tasks 7 \
    --start_task 6 &

wait
