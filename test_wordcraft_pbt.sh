# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
for trial in 1 2 3 4 5; do
    for ckpt in 40 100 200 400 1000 1500 2500 3000 3700; do
        python scripts/run_agent_v2.py \
            --llm openai/gpt-4o-mini \
            --agent_type rap_noplan \
            --num_passes 1 \
            --env wordcraft_test \
            --num_tasks 500 \
            --parallel 3 \
            --num_ic 10 \
            --start_task 0 \
            --log_name pbt_trial_${trial}_10ic_${ckpt} \
            --db_path "$CURRENT_DIR/data/wordcraft_pbt_10ic_seg10_trial_${trial}_backups/${ckpt}/learning.db/learning.db" &
        sleep 5
    done
    wait
done