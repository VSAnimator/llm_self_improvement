# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
for train_db_size in 10 40 200 800 1600 4500; do
    python scripts/run_agent_v2.py --llm openai/gpt-4o-mini --agent_type rap_noplan --num_passes 1 --env intercode_sql --num_ic 6 --num_tasks 338 --start_task 238 --db_path "$CURRENT_DIR/data/agentbank_intercode_sql_backups/${train_db_size}/learning.db/learning.db" --db_name bird_intercode_${train_db_size}_retry --parallel 2 &
done

wait