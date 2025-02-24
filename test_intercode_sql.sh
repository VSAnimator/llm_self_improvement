# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
for train_db_size in 10 15 20 40 60 100 150 200 ; do
    python scripts/run_agent_v2.py --llm openai/gpt-4o --agent_type rap_noplan --num_passes 1 --env intercode_sql --num_ic 6 --num_tasks 338 --start_task 238 --db_path "$CURRENT_DIR/data/intercode_sql_bird_backups/${train_db_size}/learning.db/learning.db" --db_name bird_gold_${train_db_size} --parallel 2 &
done

wait