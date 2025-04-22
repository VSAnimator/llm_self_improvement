# Get current directory
CURRENT_DIR=$(pwd)

# Run test script, only measure performance from 238, the rest is just for the training db
for train_db_size in 10 20 40 100 200 400; do
    python scripts/run_agent_v2.py --llm openai/gpt-4o-mini --agent_type rap_noplan --num_passes 1 --env intercode_sql --num_ic 6 --num_tasks 1038 --start_task $train_db_size --db_path "$CURRENT_DIR/data/intercode_sql_filtered/intercode_sql_gold_examples_spider_backups/${train_db_size}/learning.db/learning.db" --log_name bird_gold_${train_db_size}_cont_spider --parallel 1 --store_episodes &
    sleep 5
done

wait