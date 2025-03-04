# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
python scripts/run_agent_v2.py \
    --llm openai/gpt-4o-mini \
    --agent_type rap_refine \
    --db_path "$CURRENT_DIR/data/alfworld_expel_refine/learning.db" \
    --log_name expel_refine_v2 \
    --num_passes 10 \
    --env alfworld_test \
    --num_ic 3 \
    --store_episodes \
    --num_tasks 6 \
    --start_task 5 &

wait
