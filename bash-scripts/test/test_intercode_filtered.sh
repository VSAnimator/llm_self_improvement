# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
for trial in 1 2 3 4 5; do
    for ckpt in 40 100 200 400 800; do
        python scripts/run_agent_v2.py \
            --llm openai/gpt-4o-mini \
            --agent_type rap_noplan \
            --db_path "$CURRENT_DIR/data/intercode_sql_filtered/intercode_sql_spider_best_examples_${ckpt}/learning.db" \
            --log_name intercode_sql_spider_best_${ckpt}_trial_${trial} \
            --num_passes 1 \
            --env intercode_sql \
            --num_ic 6 \
            --parallel 10 \
            --num_tasks 1038 \
            --start_task 800
        sleep 5
    done
done

'''
for ckpt in 40 100 200 400 800; do
    python scripts/run_agent_v2.py \
        --llm openai/gpt-4o-mini \
        --agent_type rap_noplan \
        --db_path "$CURRENT_DIR/data/intercode_sql_filtered/intercode_sql_spider_worst_examples_${ckpt}/learning.db" \
        --log_name intercode_sql_spider_worst_${ckpt} \
        --num_passes 1 \
        --env intercode_sql \
        --num_ic 6 \
        --parallel 10 \
        --num_tasks 1038 \
        --start_task 800
    sleep 5
done
'''