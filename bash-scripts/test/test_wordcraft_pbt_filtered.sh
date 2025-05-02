# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
for trial in 1 2 3 4 5; do
    for ckpt in 40 100 200 400 1000 2000 4000; do
        python scripts/run_agent_v2.py \
            --llm openai/gpt-4o-mini \
            --agent_type rap_noplan \
            --num_passes 1 \
            --env wordcraft_test \
            --num_tasks 500 \
            --parallel 10 \
            --num_ic 10 \
            --start_task 0 \
            --log_name pbt_best_examples_trial_${trial}_10ic_${ckpt} \
            --db_path "$CURRENT_DIR/data/wordcraft_pbt_10_ic_best_examples_${ckpt}/learning.db"
        sleep 5
    done
done
wait