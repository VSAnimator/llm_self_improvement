# Get current directory
CURRENT_DIR=$(pwd)

python scripts/run_agent_v2.py \
    --llm openai/gpt-4o-mini \
    --agent_type rap_flex \
    --db_path "$CURRENT_DIR/data/alfworld_pbt_6ic_seg20_zero_trial_5_backups/2700/learning.db/learning.db" \
    --store_episodes \
    --env alfworld \
    --log_name pbt_6ic_seg20_zero_trial_5_2700 \
    --num_ic 6 \
    --num_tasks 3500 \
    --num_passes 1 \
    --start_task 2700 &
wait