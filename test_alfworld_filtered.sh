# Get current directory
CURRENT_DIR=$(pwd)

# Run test script

for trial in 1 2 3 4 5; do
    for ic in 6; do
        for ckpt in 40 100 200 400 1000 1500 2000 2500 3000 3500; do
            python scripts/run_agent_v2.py \
                --llm openai/gpt-4o-mini \
                --agent_type rap_flex \
                --db_path "$CURRENT_DIR/data/alfworld_filtered/alfworld_best_examples_${ic}_ic_${ckpt}/learning.db" \
                --log_name alfworld_best_examples_${ic}_ic_${ckpt}_trial_${trial} \
                --num_passes 1 \
                --env alfworld_test \
                --num_ic $ic \
                --parallel 10 \
                --num_tasks 134
            sleep 5
        done
    done
done