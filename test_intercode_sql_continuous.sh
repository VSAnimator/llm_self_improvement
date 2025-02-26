# Get current directory
CURRENT_DIR=$(pwd)

# Run test script, only measure performance from 238, the rest is just for the training db
for train_db_size in 10 15 20 40; do
    python scripts/run_agent_v2.py --llm openai/gpt-4o --agent_type rap_noplan --num_passes 1 --env intercode_sql --num_ic 6 --num_tasks 338 --start_task 40 --db_path "$CURRENT_DIR/data/intercode_sql_bird_backups/${train_db_size}_cont/learning.db/learning.db" --db_name bird_gold_${train_db_size}_cont --parallel 1 --store_episodes &
done

wait