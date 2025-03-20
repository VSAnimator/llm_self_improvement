# Get current directory
CURRENT_DIR=$(pwd)

# Run test script, only measure performance from 238, the rest is just for the training db
for train_db_size in 40; do
    python scripts/run_agent_v2.py --llm openai/gpt-4o-mini --agent_type rap_noplan --num_passes 1 --env intercode_sql --num_ic 6 --num_tasks 1000 --db_path "$CURRENT_DIR/data/intercode_sql_bird_backups/${train_db_size}/learning.db/learning.db" --log_name create_gold_spider --parallel 10 --start_task 20 &
done

wait