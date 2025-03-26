# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
for trial in 1 2 3 4 5; do
    for ckpt in 800; do
        python scripts/run_agent_v2.py \
            --llm openai/gpt-4o-mini \
            --agent_type rap_noplan \
            --num_passes 1 \
            --env intercode_sql \
            --num_ic 6 \
            --num_tasks 1038 \
            --start_task 800 \
            --log_name intercode_pbt_trial_${trial}_6ic_${ckpt}_test \
            --db_path "$CURRENT_DIR/data/intercode_sql_pbt_6ic_seg10_trial_${trial}_backups/${ckpt}/learning.db/learning.db" \
            --parallel 2 &
        sleep 5
    done
done
wait