# Get current directory
CURRENT_DIR=$(pwd)

# Run test script

for ic in 3 6; do
    for ckpt in 100 200 400 1000 1500 2000; do
        python scripts/run_agent_v2.py \
            --llm openai/gpt-4o-mini \
            --agent_type rap_flex \
            --db_path "$CURRENT_DIR/data/alfworld_filtered/alfworld_worst_examples_${ic}_ic_${ckpt}/learning.db" \
            --log_name alfworld_worst_examples_${ic}_ic_${ckpt} \
            --num_passes 1 \
            --env alfworld_test \
            --num_ic $ic \
            --parallel 10 \
            --num_tasks 134
        sleep 5
    done
    wait
done