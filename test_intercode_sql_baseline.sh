# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
for trial in 1 2 3 4 5; do
    python scripts/run_agent_v2.py \
        --llm openai/gpt-4o-mini \
        --agent_type rap_noplan \
        --db_path "$CURRENT_DIR/data/intercode_sql_spider_starter/learning.db" \
        --log_name intercode_sql_spider_baseline_trial_${trial} \
        --num_passes 1 \
        --env intercode_sql \
        --num_ic 6 \
        --parallel 10 \
        --num_tasks 1038 \
        --start_task 800
    sleep 5
done
